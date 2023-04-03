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
########       tabert        ########
### col_tabert
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
        path = '../../data/turl_cta_data/Train/' + table_title + ".json.gz"
    elif config.split == 'dev':
        path = '../../data/turl_cta_data/Validation/' + table_title + ".json.gz"
    else:
        path = '../../data/turl_cta_data/Test/' + table_title + ".json.gz"
    
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
    if other_col_num == 0:
        expected_row_num = 0
    else:
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

        if os.path.exists(f"../../data/turl_cta_data/processed_datasets/cta_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_mp{main_percentage}_col_tabert.pkl"):
            print("Loading already processed {} dataset".format(split))

            with open(f"../../data/turl_cta_data/processed_datasets/cta_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_mp{main_percentage}_col_tabert.pkl", "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/turl_cta_data/processed_datasets"))
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
                if os.path.exists("../../data/turl_cta_data/utils/common_cells.pkl"):
                    with open("../../data/turl_cta_data/utils/common_cells.pkl", "rb") as f:
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

            with open(f"../../data/turl_cta_data/processed_datasets/cta_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_mp{main_percentage}_col_tabert.pkl", 'wb') as f:
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

### CPA ### 
### col_tabert 
def cpa_tokenize_column_pair_tabert_col(column_pair_data, config):
    # input: [table name, [main col index, col index], label_ids(list)]
    table = str(column_pair_data[0])
    main_column_index = column_pair_data[1][0]
    column_index = column_pair_data[1][1]
    label = column_pair_data[2]
    
    if config.split == 'train':
        path = f'../../data/turl_cpa_data/Train/' + table + ".json.gz"
    elif config.split == 'dev':
        path = f'../../data/turl_cpa_data/Validation/' + table + ".json.gz"
    else:
        path = f'../../data/turl_cpa_data/Test/' + table + ".json.gz"
    
    df_table = pd.read_json(path, compression='gzip', lines=True)

    # prepaeing main column data
    main_col_ml = int(config.max_length * config.main_percentage)
    other_col_ml = config.max_length - main_col_ml
    other_col_num = len(df_table.columns) - 1
    # other_col_average_ml = other_col_ml // other_col_num
    
    type_func = lambda x:map_types_col(x)
    
    tokenize_func = lambda x:config.tokenizer.encode_plus(text=x,
                                                  add_special_tokens=False,
                                                  padding=True,
                                                  truncation=True,
                                                  return_attention_mask=False,
                                                  max_length=other_col_ml).input_ids
    
    df_table_typed = df_table.copy().astype("string")
    df_table_typed_temp = df_table.astype("string").apply(np.vectorize(type_func))
    df_table_typed.loc[-1, :] = df_table_typed_temp.mode().values[0]
    df_table_typed.index = df_table_typed.index +1
    df_table_typed.sort_index(inplace=True)
    
    expected_row_num = min((other_col_ml//other_col_num), df_table_typed.shape[0])  # at least 1 tokens: datatype 
    
    main_col_1 = " ".join(df_table_typed[int(main_column_index)].astype("string").to_list())
    main_col_2 = " ".join(df_table_typed[int(column_index)].astype("string").to_list())
    # two main col, using half of the main col ml
    main_col_tokens_1 = config.tokenizer.encode_plus(text=main_col_1,
                                              add_special_tokens=False,
                                              padding=True,
                                              truncation=True,
                                              return_attention_mask=False,
                                              max_length=main_col_ml//2).input_ids
    main_col_tokens_2 = config.tokenizer.encode_plus(text=main_col_2,
                                              add_special_tokens=False,
                                              padding=True,
                                              truncation=True,
                                              return_attention_mask=False,
                                              max_length=main_col_ml//2).input_ids
    
    other_col_table =  df_table_typed.drop([int(main_column_index), int(column_index)], axis=1) 
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
    
    input_ids = [config.tokenizer.cls_token_id] \
                            + main_col_tokens_1 + [config.tokenizer.sep_token_id] \
                            + main_col_tokens_2 + [config.tokenizer.sep_token_id] \
                            + other_tokens + [config.tokenizer.sep_token_id]
        
    attention_masks = [1 for _ in range(len(input_ids))] 
    assert(len(input_ids) == len(attention_masks))

    return [input_ids, attention_masks, label]

class CPAColTaBertDataset(Dataset):

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

        # cta_cta_tabert_bert-base-uncased-bs-16_ml-128_seed-42_ws-0_aug-None.pt
        if os.path.exists(f"../../data/turl_cpa_data/processed_datasets/cpa_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_mp{main_percentage}_col_tabert.pkl"):
            print("Loading already processed {} dataset".format(split))
            
            with open(f"../../data/turl_cpa_data/processed_datasets/cpa_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_mp{main_percentage}_col_tabert.pkl", "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/turl_cpa_data/processed_datasets"))
            except FileExistsError:
                pass

            #Open split
            with open(filepath, "rb") as f:
                df_dict = pickle.load(f)

            assert split in df_dict
            self.df = df_dict[split]
            self.tokenizer = tokenizer
            self.max_length = max_length-4
            self.main_percentage = main_percentage
            self.split = split
            
            #Tokenization of pairs of columns (main column + other column)
            column_pairs = [] #[main_column, other column, cpa label]

            #Main column dictionary
            main_columns = {}
            
            if (aug is not None) and (split != 'test') :
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None
            table_name = []
#             print("processing table name")
#             for index, row in self.df.loc[self.df["column_index"] != 0].iterrows():
#                 # label is on the row of non main column
#                 table_name.append([row["table_id"], row["label_ids"]])
            print("processing column index")
            non_main_table = self.df.loc[self.df["column_index"]!=0]
            # main col always 0
             # input for tokenizer [table name, [main_col index, col index], label]
            for index, row in non_main_table.iterrows():
                column_pairs.append([row["table_id"], [0, row["column_index"]], row["column_index"]])
                
            print('Processing {} {} column pairs'.format(len(column_pairs), split))
            
            # tabert will call pool in pool, so use less precesses to prevent stucking
            pool = Pool(processes=8)
            processed_cols = list(tqdm(pool.imap(partial(cpa_tokenize_column_pair_tabert_col, config=self), column_pairs, chunksize=100),total=len(column_pairs)))
            pool.close()
            pool.join()

            input_ids = []
            attention_masks = []

            #Convert to tensors
            for processed_col in processed_cols:
                input_ids.append(torch.tensor(processed_col[0]))
                attention_masks.append(torch.tensor(processed_col[1]))

            #Remove main columns
            self.df = self.df.loc[self.df["column_index"] != 0]

            self.df["data_tensor"] = input_ids
            self.df["attention_tensor"] = attention_masks
            self.df["label_tensor"] = self.df["label_ids"].apply(lambda x: torch.tensor([x]))

            with open(f"../../data/turl_cpa_data//processed_datasets/cpa_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_mp{main_percentage}_col_tabert.pkl", 'wb') as f:
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
