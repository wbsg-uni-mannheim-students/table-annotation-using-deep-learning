from functools import reduce, partial
import operator
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers
from multiprocessing import Pool
from tqdm import tqdm
import json

import re
import random
from random import choices

# From Ditto
from augment import Augmenter
from collections import Counter

# Austin 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# reng
from pandas_profiling import ProfileReport
import dateutil.parser
import datetime
import sys
sys.path.insert(0,'../..')
from util import replace_special_tokens

###############################

def collate_fn(samples):
    #Input ids: Pad to maximum batch length
    data = torch.nn.utils.rnn.pad_sequence([sample["data"] for sample in samples])
    #Labels
    label = torch.cat([sample["label"] for sample in samples])
    #Table ids
    tabs_id = [sample["table_id"] for sample in samples]
    #Attention masks: Pad to maximum batch length
    attention = torch.nn.utils.rnn.pad_sequence([sample["attention"] for sample in samples])

    # reng 
    column_indices = [sample["column_index"] for sample in samples]
    
    batch = {"data": data, "label": label, "table_id": tabs_id, "attention": attention, "column_index": column_indices}
    return batch

### CTA ###
## Austin
## method: tabert
def text_check(text):
    # at most 1 row with 512 stopwords
    text_tokens = text.split(" ")[:1024]
    if len(text_tokens) > 5:
        sw_array = np.array(stopwords.words("english"))
        tokens_without_sw = np.setdiff1d(text_tokens, sw_array, assume_unique=False) 
        text_withoud_sw = " ".join(tokens_without_sw)
        return "long_text" + " " + text_withoud_sw
    else:
        return "short_text" + " " + text
    
def map_types(cell_value):
    match_dict = {"url" : r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])",
                  "date" : r"\d{4}-[0,1]{0,1}[0-9]{0,1}-[0-3]{0,1}[0-9]{0,1}",
                  "real_number" : r"(\d+\.?\d*)"}
    
    if cell_value == " " or cell_value == "" or cell_value == "N/A" :
        return "null"
    if re.match(match_dict["url"], cell_value) is not None:
        if re.match(match_dict["url"], cell_value).group(0) == cell_value:
            return "url" + " " + cell_value
        else:
            return text_check(cell_value)

    elif re.match(match_dict["date"], cell_value) is not None:
        if re.match(match_dict["date"], cell_value).group(0) == cell_value:
            return "date" + " " + cell_value
        else:
            return text_check(cell_value)
    elif re.match(match_dict["real_number"], cell_value) is not None:
        if re.match(match_dict["real_number"], cell_value).group(0) == cell_value:
            return "real_number" + " " + cell_value
        else:
            return text_check(cell_value)
    else:
        return text_check(cell_value)

def cta_tokenize_tabert(column_data, config):
    column_index = column_data[0]
    table_title = column_data[1]

    if config.split == 'train':
        path = '../../data/CTA/{}/Train/'.format(config.folder) + table_title
    elif config.split == 'dev':
        path = '../../data/CTA/{}/Validation/'.format(config.folder) + table_title
    else:
        path = '../../data/CTA/{}/Test/'.format(config.folder) + table_title
    
    df_table = pd.read_json(path, compression='gzip', lines=True)
    
    max_length = config.max_length - 2 # CLS and SEP
    main_col_ml = int(max_length * config.main_percentage)
    
    type_func = lambda x:map_types(x)
    df_table_typed = df_table.astype("string").apply(np.vectorize(type_func))
    main_col = " ".join(df_table_typed[int(column_index)].astype("string").to_list())
    main_col_tokens = config.tokenizer.encode_plus(text=main_col,
                                                  add_special_tokens=False,
                                                  padding=True,
                                                  truncation=True,
                                                  return_attention_mask=False,
                                                  max_length=main_col_ml).input_ids

    other_col_ml = max_length - len(main_col_tokens)
    other_col_num = len(df_table.columns) - 1    
    expected_row_num = min((other_col_ml//other_col_num), df_table_typed.shape[0])  # at least 1 tokens: datatype 
    
    other_col_table =  df_table_typed.drop(int(column_index), axis=1).truncate(after=expected_row_num)
    others = []
    
    for row in range(expected_row_num):
        others.append('[SEP]')
        row_data = " ".join(other_col_table.loc[row,:].tolist())
        others.append(row_data)
    others = " ".join(others)
    other_tokens = config.tokenizer.encode_plus(text=others,
                                                  add_special_tokens=False,
                                                  padding=True,
                                                  truncation=True,
                                                  return_attention_mask=False,
                                                  max_length=other_col_ml).input_ids
    input_ids = [config.tokenizer.cls_token_id] + main_col_tokens + other_tokens + [config.tokenizer.sep_token_id]
    attention_mask = [1 for _ in range(len(input_ids))] 
    
    assert(len(input_ids) == len(attention_mask))
    
    new_dict = {}
    for i in ('input_ids', 'attention_mask'):
        new_dict[i] = locals()[i]
    
    return new_dict

class CTATaBertDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 bert: str = 'bert' ,
                 window_size: int = 0,
                 main_percentage: float = 0.5,
                 device: torch.device = None,
                 aug=None,
                 preprocess=None):

        if device is None:
            device = torch.device('cpu')

        if (preprocess=='tuta'):
            self.folder = 'TUTA_FULL'
        elif (preprocess=='median'):
            self.folder = 'MEDIAN_FULL'
        elif (preprocess=='mean'): 
            self.folder = 'MEAN_FULL'
        else:
            self.folder = 'NEW_FULL'
        folder = self.folder

        if os.path.exists(f"../../data/CTA/{folder}/processed_datasets/cta_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_mp{main_percentage}_tabert.pkl"):
            print("Loading already processed {} dataset".format(split))

            with open(f"../../data/CTA/{folder}/processed_datasets/cta_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_mp{main_percentage}_tabert.pkl", "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/CTA/{}/processed_datasets".format(folder)))
            except FileExistsError:
                pass

            with open(filepath, "rb") as f:
                df_dict = pickle.load(f)

            assert split in df_dict
            self.df = df_dict[split]
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.main_percentage = main_percentage
            self.split = split
            if (aug is not None) and (split != 'test'):
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None
            self.mp = main_percentage

            cols_index = self.df["column_index"].tolist()
            cols_table_id = self.df["table_id"].tolist()

            cols = [list(x) for x in zip(cols_index, cols_table_id)]

            if (self.aug == 'freq_cell_sampling'):
                if os.path.exists("../../data/CTA/{}/utils/common_cells.pkl".format(folder)):
                    with open("../../data/CTA/{}/utils/common_cells.pkl".format(folder), "rb") as f:
                        self.common_cells = pickle.load(f)[split]

                    print("Loaded common cells")
                else:
                    print("No common cells uploaded")
                    raise SystemExit

            print('Processing {} {} columns'.format(len(cols), split))

            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(cta_tokenize_tabert, config=self), cols, chunksize=100),total=len(cols)))
            pool.close()
            pool.join()

            input_ids = []
            attention_masks = []

            for processed_col in processed_cols:
                input_ids.append(torch.tensor(processed_col["input_ids"]))
                attention_masks.append(torch.tensor(processed_col["attention_mask"]))

            self.df["data_tensor"] = input_ids
            self.df["attention_tensor"] = attention_masks
            self.df["label_tensor"] = self.df["label_ids"].apply(lambda x: torch.tensor([x]))

            with open(f"../../data/CTA/{folder}/processed_datasets/cta_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_mp{main_percentage}_tabert.pkl", 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"],
            "column_index": self.df.iloc[idx]["column_index"]
        }

### method: col_tabert
def text_check_col(text):
    # at most 1 row with 512 stopwords
    text_tokens = text.split(" ")[:1024]
    if len(text_tokens) > 5:
        return "long_text"
    else:
        return "short_text"
    
def map_types_col(cell_value):
    match_dict = {"url" : r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])",
                  "date" : r"\d{4}-[0,1]{0,1}[0-9]{0,1}-[0-3]{0,1}[0-9]{0,1}",
                  "real_number" : r"(\d+\.?\d*)"}
    
    if cell_value == " " or cell_value == "" or cell_value == "N/A" :
        return "null"
    if re.match(match_dict["url"], cell_value) is not None:
        if re.match(match_dict["url"], cell_value).group(0) == cell_value:
            return "url"
        else:
            return text_check_col(cell_value)

    elif re.match(match_dict["date"], cell_value) is not None:
        if re.match(match_dict["date"], cell_value).group(0) == cell_value:
            return "date"
        else:
            return text_check_col(cell_value)
    elif re.match(match_dict["real_number"], cell_value) is not None:
        if re.match(match_dict["real_number"], cell_value).group(0) == cell_value:
            return "real_number"
        else:
            return text_check_col(cell_value)
    else:
        return text_check_col(cell_value)
    
def cta_tokenize_tabert_col(column_data, config):
    column_index = column_data[0]
    table_title = column_data[1]

    if config.split == 'train':
        path = '../../data/CTA/{}/Train/'.format(config.folder) + table_title
    elif config.split == 'dev':
        path = '../../data/CTA/{}/Validation/'.format(config.folder) + table_title
    else:
        path = '../../data/CTA/{}/Test/'.format(config.folder) + table_title
    
    df_table = pd.read_json(path, compression='gzip', lines=True)
    
    max_length = config.max_length - 2 # CLS and SEP
    main_col_ml = int(max_length * config.main_percentage)
    
    type_func = lambda x:map_types_col(x)
    df_table_typed = df_table.astype("string").apply(np.vectorize(type_func))
    df_table.loc[-1, :] = df_table_typed.mode().values[0]
    df_table.index = df_table.index +1
    df_table.sort_index(inplace=True)

    main_col = " ".join(df_table[int(column_index)].astype("string").to_list())
    main_col_tokens = config.tokenizer.encode_plus(text=main_col,
                                                  add_special_tokens=False,
                                                  padding=True,
                                                  truncation=True,
                                                  return_attention_mask=False,
                                                  max_length=main_col_ml).input_ids

    other_col_ml = max_length - len(main_col_tokens)
    other_col_num = len(df_table.columns) - 1    
    expected_row_num = min((other_col_ml//other_col_num), df_table_typed.shape[0])  # at least 1 tokens: datatype 
    
    other_col_table =  df_table_typed.drop(int(column_index), axis=1).truncate(after=expected_row_num)
    others = []
    
    for row in range(expected_row_num):
        others.append('[SEP]')
        row_data = " ".join(other_col_table.loc[row,:].tolist())
        others.append(row_data)
    others = " ".join(others)
    other_tokens = config.tokenizer.encode_plus(text=others,
                                                  add_special_tokens=False,
                                                  padding=True,
                                                  truncation=True,
                                                  return_attention_mask=False,
                                                  max_length=other_col_ml).input_ids
    input_ids = [config.tokenizer.cls_token_id] + main_col_tokens + other_tokens + [config.tokenizer.sep_token_id]
    attention_mask = [1 for _ in range(len(input_ids))] 
    
    assert(len(input_ids) == len(attention_mask))
    
    new_dict = {}
    for i in ('input_ids', 'attention_mask'):
        new_dict[i] = locals()[i]
    
    return new_dict

class CTAColTaBertDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 bert: str = 'bert' ,
                 window_size: int = 0,
                 main_percentage: float = 0.5,
                 device: torch.device = None,
                 aug=None,
                 preprocess=None):

        if device is None:
            device = torch.device('cpu')

        if (preprocess=='tuta'):
            self.folder = 'TUTA_FULL'
        elif (preprocess=='median'):
            self.folder = 'MEDIAN_FULL'
        elif (preprocess=='mean'): 
            self.folder = 'MEAN_FULL'
        else:
            self.folder = 'NEW_FULL'
        folder = self.folder

        if os.path.exists(f"../../data/CTA/{folder}/processed_datasets/cta_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_mp{main_percentage}_col_tabert.pkl"):
            print("Loading already processed {} dataset".format(split))

            with open(f"../../data/CTA/{folder}/processed_datasets/cta_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_mp{main_percentage}_col_tabert.pkl", "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/CTA/{}/processed_datasets".format(folder)))
            except FileExistsError:
                pass

            with open(filepath, "rb") as f:
                df_dict = pickle.load(f)

            assert split in df_dict
            self.df = df_dict[split]
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.main_percentage = main_percentage
            self.split = split
            if (aug is not None) and (split != 'test'):
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None
            self.mp = main_percentage
            
            cols_index = self.df["column_index"].tolist()
            cols_table_id = self.df["table_id"].tolist()

            cols = [list(x) for x in zip(cols_index, cols_table_id)]

            if (self.aug == 'freq_cell_sampling'):
                if os.path.exists("../../data/CTA/{}/utils/common_cells.pkl".format(folder)):
                    with open("../../data/CTA/{}/utils/common_cells.pkl".format(folder), "rb") as f:
                        self.common_cells = pickle.load(f)[split]

                    print("Loaded common cells")
                else:
                    print("No common cells uploaded")
                    raise SystemExit

            print('Processing {} {} columns'.format(len(cols), split))

            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(cta_tokenize_tabert_col, config=self), cols, chunksize=100),total=len(cols)))
            pool.close()
            pool.join()

            input_ids = []
            attention_masks = []

            for processed_col in processed_cols:
                input_ids.append(torch.tensor(processed_col["input_ids"]))
                attention_masks.append(torch.tensor(processed_col["attention_mask"]))

            self.df["data_tensor"] = input_ids
            self.df["attention_tensor"] = attention_masks
            self.df["label_tensor"] = self.df["label_ids"].apply(lambda x: torch.tensor([x]))

            with open(f"../../data/CTA/{folder}/processed_datasets/cta_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_mp{main_percentage}_col_tabert.pkl", 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"],
            "column_index": self.df.iloc[idx]["column_index"]
        }


## Reng
### method: sum_neighbor
def cta_tokenize_sum_neighbor_columns(column_data, config):
    max_len_avail = config.max_length - (2 + config.window_size * 2)
    max_len_other_col = int(max_len_avail * (1 - config.main_percentage) // (config.window_size * 2))
    max_len_target_col = max_len_avail - int(max_len_other_col * (config.window_size * 2))

    column_index = column_data[0]
    table_title = column_data[1]

    if config.split == 'train':
        path = '../../data/CTA/{}/Train/'.format(config.folder) + table_title
    elif config.split == 'dev':
        path = '../../data/CTA/{}/Validation/'.format(config.folder) + table_title
    else:
        path = '../../data/CTA/{}/Test/'.format(config.folder) + table_title

    df_table = pd.read_json(path, compression='gzip', lines=True)
    df_statistics = config.statistics
    df_table_statistics = df_statistics.loc[df_statistics['table_title'] == table_title]
    assert len(df_table_statistics) == len(df_table.columns)

    target_col_stats = df_table_statistics.loc[df_table_statistics.column_index == str(column_index)].values[0][2]
    target_col_stats = " ".join(replace_special_tokens(target_col_stats).split())
    
    target_col_list = df_table[int(column_index)].astype("string").tolist()
    
    if config.aug != None:
        target_col_list = Augmenter.augment(target_col_list, config=config)
        
    target_col_data = target_col_stats + " [VAL] " + " ".join(target_col_list)     
    target_col_token = config.tokenizer.encode_plus(target_col_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_target_col).input_ids
    
    input_ids = [config.tokenizer.cls_token_id] + target_col_token + [config.tokenizer.sep_token_id]
    
    for i in range(1, (config.window_size + 1)):
        left_index = ((int(column_index) - i) % len(df_table.columns))
        right_index = ((int(column_index) + i) % len(df_table.columns))

        left_index_stats = df_table_statistics.loc[df_table_statistics.column_index == str(left_index)].values[0][2]
        left_index_stats = " ".join(replace_special_tokens(left_index_stats).split())
        left_data_list = df_table[int(left_index)].astype("string").tolist()
        if config.aug != None:
            left_data_list = Augmenter.augment(left_data_list, config=config)
        left_data = left_index_stats + " [VAL] " + " ".join(left_data_list)

        right_index_stats = df_table_statistics.loc[df_table_statistics.column_index == str(right_index)].values[0][2]
        right_index_stats = " ".join(replace_special_tokens(right_index_stats).split())
        right_data_list = df_table[int(right_index)].astype("string").tolist()
        if config.aug != None:
            right_data_list = Augmenter.augment(right_data_list, config=config)
        right_data = right_index_stats + " [VAL] " + " ".join(right_data_list)

        left_token = config.tokenizer.encode_plus(left_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_other_col).input_ids
        right_token = config.tokenizer.encode_plus(right_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_other_col).input_ids
        input_ids = input_ids + left_token + [config.tokenizer.sep_token_id] + right_token + [config.tokenizer.sep_token_id]

    attention_mask = [1 for _ in range(len(input_ids))]
    assert(len(input_ids) == len(attention_mask))
    
    new_dict = {}
    for i in ('input_ids', 'attention_mask'):
        new_dict[i] = locals()[i]

    return new_dict

class CTASumNeighborColumnDataset(Dataset):
    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 window_size: int = 0, #0/1/2
                 bert: str = 'bert' ,
                 device: torch.device = None,
                 aug=None,
                 preprocess=None, 
                 main_percentage: float = 0.5):

        if device is None:
            device = torch.device('cpu')

        if (preprocess=='tuta'):
            self.folder = 'TUTA_FULL'
        elif (preprocess=='median'):
            self.folder = 'MEDIAN_FULL'
        elif (preprocess=='mean'): 
            self.folder = 'MEAN_FULL'
        else:
            self.folder = 'NEW_FULL'
        folder = self.folder
        
        # _neighbor
        if os.path.exists("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_mp{}_sum_neighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug, main_percentage)):
            print("Loading already processed {} dataset".format(split))

            with open("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_mp{}_sum_neighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug, main_percentage), "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/CTA/{}/processed_datasets".format(folder)))
            except FileExistsError:
                pass

            #Open split
            with open(filepath, "rb") as f:
                df_dict = pickle.load(f)

            assert split in df_dict
            self.df = df_dict[split]
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.window_size = window_size
            self.main_percentage = main_percentage
            self.split = split
            if (aug is not None) and (split != 'test') :
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None

            cols_index = self.df["column_index"].tolist()
            cols_table_id = self.df["table_id"].tolist()

            stat_path = '../cta/{}_dict.pkl'.format(split)
            with open(stat_path, "rb") as f:
                stats = pickle.load(f)
            self.statistics = pd.DataFrame(stats, columns=['table_title', 'column_index', 'sum_data'])

            cols = [list(x) for x in zip(cols_index, cols_table_id)]

            print('Processing {} {} columns'.format(len(cols), split))

            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(cta_tokenize_sum_neighbor_columns, config=self), cols, chunksize=100),total=len(cols)))
            pool.close()
            pool.join()

            input_ids = []
            attention_masks = []

            for processed_col in processed_cols:
                input_ids.append(torch.tensor(processed_col["input_ids"]))
                attention_masks.append(torch.tensor(processed_col["attention_mask"]))

            self.df["data_tensor"] = input_ids
            self.df["attention_tensor"] = attention_masks
            self.df["label_tensor"] = self.df["label_ids"].apply(lambda x: torch.tensor([x]))

            with open("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_mp{}_sum_neighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug, main_percentage), 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"],
            "column_index": self.df.iloc[idx]["column_index"]
        }

## method: neighbor
def cta_tokenize_neighbor_columns(column_data, config):
    max_len_avail = int(config.max_length) - (2 + int(config.window_size) * 2)
    max_len_other_col = int(max_len_avail * (1 - config.main_percentage) // (config.window_size * 2))
    max_len_main_col = max_len_avail - int(max_len_other_col * (config.window_size * 2))
    column_index = column_data[0]
    table_title = column_data[1]

    if config.split == 'train':
        path = '../../data/CTA/{}/Train/'.format(config.folder) + table_title
    elif config.split == 'dev':
        path = '../../data/CTA/{}/Validation/'.format(config.folder) + table_title
    else:
        path = '../../data/CTA/{}/Test/'.format(config.folder) + table_title

    df_table = pd.read_json(path, compression='gzip', lines=True)

    column_data_list = df_table[int(column_index)].astype("string").tolist()
    if config.aug != None:
        column_data_list = Augmenter.augment(column_data_list, config=config)
    target_data = " ".join(column_data_list)

    col_token = config.tokenizer.encode_plus(target_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_main_col).input_ids
    input_ids = [config.tokenizer.cls_token_id] + col_token + [config.tokenizer.sep_token_id]

    for i in range(1, (config.window_size + 1)):
        left_index = ((int(column_index) - i) % len(df_table.columns))
        right_index = ((int(column_index) + i) % len(df_table.columns))

        left_data_list = df_table[int(left_index)].astype("string").tolist()
        if config.aug != None:
            left_data_list = Augmenter.augment(left_data_list, config=config)
        left_data = " ".join(left_data_list)

        right_data_list = df_table[int(right_index)].astype("string").tolist()
        if config.aug != None:
            right_data_list = Augmenter.augment(right_data_list, config=config)
        right_data = " ".join(right_data_list)

        left_token = config.tokenizer.encode_plus(left_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_other_col).input_ids
        right_token = config.tokenizer.encode_plus(right_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_other_col).input_ids

        input_ids = input_ids + left_token + [config.tokenizer.sep_token_id] + right_token + [config.tokenizer.sep_token_id]

    attention_mask = [1 for _ in range(len(input_ids))]
    assert(len(input_ids) == len(attention_mask))

    new_dict = {}
    for i in ('input_ids', 'attention_mask'):
        new_dict[i] = locals()[i]

    return new_dict

class CTANeighborColumnDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 window_size: int = 0, #0/1/2
                 bert: str = 'bert' ,
                 device: torch.device = None,
                 aug=None,
                 preprocess=None, 
                 main_percentage: float = 0.5):

        if device is None:
            device = torch.device('cpu')

        if (preprocess=='tuta'):
            self.folder = 'TUTA_FULL'
        elif (preprocess=='median'):
            self.folder = 'MEDIAN_FULL'
        else:
            self.folder = 'NEW_FULL'
        folder = self.folder

        if os.path.exists("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_mp{}_neighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug, main_percentage)):
            print("Loading already processed {} dataset".format(split))

            with open("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_mp{}_neighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug, main_percentage), "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/CTA/{}/processed_datasets".format(folder)))
            except FileExistsError:
                pass

            #Open split
            with open(filepath, "rb") as f:
                df_dict = pickle.load(f)

            assert split in df_dict
            self.df = df_dict[split]
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.window_size = window_size
            self.main_percentage = main_percentage
            self.split = split
            if (aug is not None) and (split != 'test') :
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None


            cols_index = self.df["column_index"].tolist()
            cols_table_id = self.df["table_id"].tolist()

            cols = [list(x) for x in zip(cols_index, cols_table_id)]

            print('Processing {} {} columns'.format(len(cols), split))

            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(cta_tokenize_neighbor_columns, config=self), cols, chunksize=100),total=len(cols)))
            pool.close()
            pool.join()

            input_ids = []
            attention_masks = []

            for processed_col in processed_cols:
                input_ids.append(torch.tensor(processed_col["input_ids"]))
                attention_masks.append(torch.tensor(processed_col["attention_mask"]))

            self.df["data_tensor"] = input_ids
            self.df["attention_tensor"] = attention_masks
            self.df["label_tensor"] = self.df["label_ids"].apply(lambda x: torch.tensor([x]))

            with open("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_mp{}_neighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug, main_percentage), 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"],
            "column_index": self.df.iloc[idx]["column_index"]
        }

## end Austin/Reng 
## single_col
def cta_tokenize_columns(column_data, config):
    max_len_col = config.max_length - 2
    column_index = column_data[0]
    table_title = column_data[1]

    if config.split == 'train':
        path = '../../data/CTA/{}/Train/'.format(config.folder) + table_title
    elif config.split == 'dev':
        path = '../../data/CTA/{}/Validation/'.format(config.folder) + table_title
    else:
        path = '../../data/CTA/{}/Test/'.format(config.folder) + table_title

    df_table = pd.read_json(path, compression='gzip', lines=True)

    column_data_list = df_table[int(column_index)].astype("string").tolist()
    
    if config.aug != None:
        column_data_list = Augmenter.augment(column_data_list, config=config)
    target_data = " ".join(column_data_list)

    col_token = config.tokenizer.encode_plus(target_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_col).input_ids
    input_ids = [config.tokenizer.cls_token_id] + col_token + [config.tokenizer.sep_token_id]

    attention_mask = [1 for _ in range(len(input_ids))]
    assert(len(input_ids) == len(attention_mask))

    new_dict = {}
    for i in ('input_ids', 'attention_mask'):
        new_dict[i] = locals()[i]

    return new_dict

class CTASingleColumnDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 window_size: int = 0, #0/1/2
                 bert: str = 'bert' ,
                 device: torch.device = None,
                 aug=None,
                 preprocess=None,
                main_percentage: float = 0.5):

        if device is None:
            device = torch.device('cpu')

        if (preprocess=='tuta'):
            self.folder = 'TUTA_FULL'
        elif (preprocess=='median'):
            self.folder = 'MEDIAN_FULL'
        else:
            self.folder = 'NEW_FULL'
        folder = self.folder

        if os.path.exists("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_mp{}_singlecolumn.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug, main_percentage)):
            print("Loading already processed {} dataset".format(split))

            with open("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_mp{}_singlecolumn.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug, main_percentage), "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/CTA/{}/processed_datasets".format(folder)))
            except FileExistsError:
                pass

            #Open split
            with open(filepath, "rb") as f:
                df_dict = pickle.load(f)

            assert split in df_dict
            self.df = df_dict[split]
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.split = split
            self.main_percentage = main_percentage
            if (aug is not None) and (split != 'test') :
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None

            cols_index = self.df["column_index"].tolist()
            cols_table_id = self.df["table_id"].tolist()

            cols = [list(x) for x in zip(cols_index, cols_table_id)]

            print('Processing {} {} columns'.format(len(cols), split))

            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(cta_tokenize_columns, config=self), cols, chunksize=100),total=len(cols)))
            pool.close()
            pool.join()

            input_ids = []
            attention_masks = []

            for processed_col in processed_cols:
                input_ids.append(torch.tensor(processed_col["input_ids"]))
                attention_masks.append(torch.tensor(processed_col["attention_mask"]))

            self.df["data_tensor"] = input_ids
            self.df["attention_tensor"] = attention_masks
            self.df["label_tensor"] = self.df["label_ids"].apply(lambda x: torch.tensor([x]))

            with open("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_mp{}_singlecolumn.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug, main_percentage), 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"],
            "column_index": self.df.iloc[idx]["column_index"]
        }