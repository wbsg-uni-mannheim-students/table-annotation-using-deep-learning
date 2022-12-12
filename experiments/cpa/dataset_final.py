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
from nltk.tokenize import word_tokenize

# reng
from pandas_profiling import ProfileReport
import dateutil.parser
import datetime

##### Munir ############
from collections import defaultdict, Counter
import operator
from heapq import nlargest
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
import string
from num2words import num2words

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

    batch = {"data": data, "label": label, "table_id": tabs_id, "attention": attention}
    return batch

#COLUMN TYPE ANNOTATION

# Basic Serialization: Convert single column to sequence
# [CLS] column value [SEP]
def tokenize_columns(column_data, config):
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

        if os.path.exists("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_singlecolumn.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug)):
            print("Loading already processed {} dataset".format(split))

            with open("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_singlecolumn.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), "rb") as f:
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
            if (aug is not None) and (split != 'test') :
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None

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
            processed_cols = list(tqdm(pool.imap(partial(tokenize_columns, config=self), cols, chunksize=100),total=len(cols)))
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

            with open("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_singlecolumn.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"]
        }

#Table serialization as in DODUO
# [CLS] column 1 value [SEP] [CLS] column 2 value [SEP] ...
#Drawback: max_column needs to be specified
def tokenize_all_table(group_df, config):
    #Retrieve grouped columns of same table:
    group_df = group_df[1]

    #Constraint for table columns: skip tables with more than max_column columns
    if len(group_df) > config.max_colnum:
        return []

    #Tokenize columns
    column_values = group_df["data"].tolist()
    #Encode each column value separately
    encoded_columns = [ config.tokenizer.encode_plus( x[:200000], add_special_tokens=True, truncation=True, padding = True, return_attention_mask = True, max_length=config.max_length + 2) for x in column_values ]

    token_ids_list = [ column["input_ids"] for column in encoded_columns]
    attention_masks_list = [column["attention_mask"] for column in encoded_columns]

    #Concatenate all encoded columns and attention masks together
    token_ids = reduce(operator.add, token_ids_list)
    attention_masks = reduce(operator.add, attention_masks_list)

    #List with the indices where the CLS tokens are located (the length of each column helps find where the CLS tokens are)
    cls_index_list = [0] + np.cumsum(np.array([len(x) for x in token_ids_list])).tolist()[:-1]

    #Check if the indices listed belong to the CLS token
    for cls_index in cls_index_list:
        assert token_ids[cls_index] == config.tokenizer.cls_token_id, "cls_indexes validation"

    return [group_df['table_id'].tolist()[0], len(group_df), token_ids, group_df["label_ids"].tolist(), cls_index_list, attention_masks]


class CTAAllTableDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 15,
                 bert: str = 'bert' ,
                 device: torch.device = None):

        if device is None:
            device = torch.device('cpu')

        if os.path.exists("../../data/CTA/processed_datasets/cta_{}_{}_{}_alltable.pkl".format(folder, split, bert, max_length)):
            print("Loading already processed {} dataset".format(split))
            with open("../../data/CTA/processed_datasets/cta_{}_{}_{}_alltable.pkl".format(folder, split, bert, max_length), "rb") as f:
                df_dict = pickle.load(f)
            self.table_df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/CTA/processed_datasets"))
            except FileExistsError:
                pass

            with open(filepath, "rb") as fin:
                df_dict = pickle.load(fin)

            assert split in df_dict
            self.df = df_dict[split]
            num_tables = len(self.df.groupby("table_id"))

            self.tokenizer = tokenizer
            self.max_length = max_length
            self.max_colnum = max_colnum

            #Tokenization
            print('Processing {} {} tables'.format(num_tables, split))

            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(tokenize_all_table, config = self), self.df.groupby("table_id"), chunksize=100), total=num_tables))
            pool.close()
            pool.join()

            #Filter out tables with more than max_colnum columns
            data_list = [x for x in processed_cols if x]
            print("{} tables with less than {} columns".format(len(data_list), max_colnum))


            self.table_df = pd.DataFrame(data_list,
                                         columns=[
                                             "table_id", "num_col", "data_encoded",
                                             "label_list", "cls_indexes_list", "attention_mask"
                                         ])

            #Convert to tensors
            self.table_df["data_tensor"] = self.table_df["data_encoded"].apply(lambda x: torch.LongTensor(x).to(device))
            self.table_df["label_tensor"] = self.table_df["label_list"].apply(lambda x: torch.LongTensor(x).to(device))
            self.table_df["cls_indexes"] = self.table_df["cls_indexes_list"].apply(lambda x: torch.LongTensor(x).to(device))
            self.table_df["attention_tensor"] = self.table_df["attention_mask"].apply(lambda x: torch.LongTensor(x).to(device))

            with open("../../data/CTA/processed_datasets/cta_{}_{}_{}_alltable.pkl".format(folder, split, bert, max_length), 'wb') as f:
                pickle.dump(self.table_df, f)


    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "attention": self.table_df.iloc[idx]["attention_tensor"],
            "table_id": self.table_df.iloc[idx]["table_id"]
        }

#COLUMNS PAIR ANNOTATION

#Basic serialization: Main column + [SEP] + Other column
def tokenize_column_pair(column_pair_data, config):
    max_len = config.max_length - 3 # [CLS] + Main_Col + [SEP] + Target_Col [SEP] 
    column_index = column_pair_data[0]
    table_title = column_pair_data[1]
    main_col_len = max_len // 2 # 254 since the target col should be more important 
    target_col_len = max_len - main_col_len # 255 
    
    if config.split == 'train':
        path = '../../data/CPA/{}/Train/'.format(config.folder) + table_title
    elif config.split == 'dev':
        path = '../../data/CPA/{}/Validation/'.format(config.folder) + table_title
    else:
        path = '../../data/CPA/{}/Test/'.format(config.folder) + table_title

    df_table = pd.read_json(path, compression='gzip', lines=True)

    main_col_list = df_table[int('0')].astype("string").tolist()
    target_col_list = df_table[int(column_index)].astype("string").tolist()
    
    if config.aug != None:
        main_col_list = Augmenter.augment(main_col_list, config=config)
        target_col_list = Augmenter.augment(target_col_list, config=config)
        
    main_col_data = " ".join(main_col_list)
    target_col_data = " ".join(target_col_list) 
    
    main_col_token = config.tokenizer.encode_plus(main_col_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=main_col_len).input_ids
    target_col_token = config.tokenizer.encode_plus(target_col_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=target_col_len).input_ids
    
    input_ids = [config.tokenizer.cls_token_id] + main_col_token + [config.tokenizer.sep_token_id] + target_col_token + [config.tokenizer.sep_token_id]

    attention_mask = [1 for _ in range(len(input_ids))]
    assert(len(input_ids) == len(attention_mask))

    new_dict = {}
    for i in ('input_ids', 'attention_mask'):
        new_dict[i] = locals()[i]

    return new_dict

class CPASingleColumnDataset(Dataset):
    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 window_size: int = 0, #0/1/2
                 bert: str = 'bert' ,
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

        if os.path.exists("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_singlecolumn.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug)):
            print("Loading already processed {} dataset".format(split))

            with open("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_singlecolumn.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/CPA/{}/processed_datasets".format(folder)))
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
            if (aug is not None) and (split == 'train') :
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None
                
            cols_index = self.df["column_index"].tolist()
            cols_table_id = self.df["table_id"].tolist()

            cols = [list(x) for x in zip(cols_index, cols_table_id)]

            print('Processing {} {} column pairs'.format(len(cols), split))
            # print(tokenize_column_pair(cols[1], config=self))

            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(tokenize_column_pair, config=self), cols, chunksize=100),total=len(cols)))
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

            with open("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_singlecolumn.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"],
        }

#Table serialization as in DODUO
# [CLS] column 1 value [SEP] [CLS] column 2 value [SEP] ...
#Drawback: max_column needs to be specified
def tokenize_all_table_cpa(group_df, config):

    #Retrieve grouped columns of same table:
    group_df = group_df[1]

    #Constraint for table columns: skip tables with more than max_column columns
    if len(group_df) > config.max_colnum:
        return []

    group_df['column_id'] = group_df['column_id'].astype(str)

    #Sort by column index
    group_df = group_df.sort_values("column_id")

    #Tokenize columns
    column_values = group_df["data"].tolist()
    #Encode each column value separately
    encoded_columns = [ config.tokenizer.encode_plus( x[:200000], add_special_tokens=True, truncation=True, padding = True, return_attention_mask = True, max_length=config.max_length + 2) for x in column_values ]

    token_ids_list = [ column["input_ids"] for column in encoded_columns]
    attention_masks_list = [column["attention_mask"] for column in encoded_columns]

    #Concatenate all encoded columns and attention masks together
    token_ids = reduce(operator.add, token_ids_list)
    attention_masks = reduce(operator.add, attention_masks_list)

    #List with the indices where the CLS tokens are located (the length of each column helps find where the CLS tokens are)
    cls_index_list = [0] + np.cumsum(np.array([len(x) for x in token_ids_list])).tolist()[:-1]

    #Check if the indices listed belong to the CLS token
    for cls_index in cls_index_list:
        assert token_ids[cls_index] == config.tokenizer.cls_token_id, "cls_indexes validation"

    return [group_df['table_id'].tolist()[0], len(group_df), token_ids, group_df["label_ids"].tolist(), cls_index_list, attention_masks, cls_index_list]


class CPAAllTableDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 15, #10
                 bert: str = 'bert' ,
                 device: torch.device = None):

        if device is None:
            device = torch.device('cpu')

        if os.path.exists("data/CPA/processed_datasets/cpa_{}_{}_{}_alltable.pkl".format(folder, split, bert, max_length)):
            print("Loading already processed {} dataset".format(split))
            with open("data/CPA/processed_datasets/cpa_{}_{}_{}_alltable.pkl".format(folder, split, bert, max_length), "rb") as f:
                df_dict = pickle.load(f)
            self.table_df = df_dict

        else:
            try:
                os.mkdir(os.path.join("data/CPA/processed_datasets"))
            except FileExistsError:
                pass

            with open(filepath, "rb") as fin:
                df_dict = pickle.load(fin)

            assert split in df_dict
            self.df = df_dict[split]
            num_tables = len(self.df.groupby("table_id"))

            self.tokenizer = tokenizer
            self.max_length = max_length
            self.max_colnum = max_colnum

            #Tokenization
            print('Processing {} {} tables'.format(num_tables, split))

            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(tokenize_all_table_cpa, config = self), self.df.groupby("table_id"), chunksize=100), total=num_tables))
            pool.close()
            pool.join()

            #Filter out tables with more than max_colnum columns
            data_list = [x for x in processed_cols if x]
            print("{} tables with less than {} columns".format(len(data_list), max_colnum))


            self.table_df = pd.DataFrame(data_list,
                                         columns=[
                                             "table_id", "num_col", "data_encoded",
                                             "label_list", "cls_indexes_list", "attention_mask", "cls_index_list"
                                         ])

            #Convert to tensors
            self.table_df["data_tensor"] = self.table_df["data_encoded"].apply(lambda x: torch.LongTensor(x).to(device))
            self.table_df["label_tensor"] = self.table_df["label_list"].apply(lambda x: torch.LongTensor(x).to(device))
            self.table_df["cls_indexes"] = self.table_df["cls_indexes_list"].apply(lambda x: torch.LongTensor(x).to(device))
            self.table_df["attention_tensor"] = self.table_df["attention_mask"].apply(lambda x: torch.LongTensor(x).to(device))

            with open("data/CPA/processed_datasets/cpa_{}_{}_{}_alltable.pkl".format(folder, split, bert, max_length), 'wb') as f:
                pickle.dump(self.table_df, f)


    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "attention": self.table_df.iloc[idx]["attention_tensor"],
            "table_id": self.table_df.iloc[idx]["table_id"],
        }

######################################
# RENG
def tokenize_randomneighbor_columns(column_data, config):
    max_len_avail = config.max_length - (2 + config.window_size * 2)
    max_len_other_col = max_len_avail // (1 + config.window_size * 2)
    max_len_main_col = max_len_avail - (max_len_other_col * config.window_size * 2)

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

    table_length = len(df_table.columns)
    remaining_index = list(filter(lambda x: x!= column_index, range(table_length)))

    num_to_pick = config.window_size * 2

    if len(remaining_index) < num_to_pick:
        index_to_use = (choices(remaining_index, k = num_to_pick))
    else:
        index_to_use = (random.sample(remaining_index, num_to_pick))

    for i in index_to_use:
        neighbor_data_list = df_table[int(i)].astype("string").tolist()
        if config.aug != None:
            neighbor_data_list = Augmenter.augment(neighbor_data_list, config=config)
        neighbor_data = " ".join(neighbor_data_list)

        neighbor_token = config.tokenizer.encode_plus(neighbor_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_other_col).input_ids

        input_ids = input_ids + neighbor_token + [config.tokenizer.sep_token_id]

    attention_mask = [1 for _ in range(len(input_ids))]
    assert(len(input_ids) == len(attention_mask))

    new_dict = {}
    for i in ('input_ids', 'attention_mask'):
        new_dict[i] = locals()[i]

    return new_dict

class CTARandomColumnDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 window_size: int = 0, #0/1/2
                 bert: str = 'bert' ,
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


        if os.path.exists("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_randomneighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug)):
            print("Loading already processed {} dataset".format(split))

            with open("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_randomneighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), "rb") as f:
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
            processed_cols = list(tqdm(pool.imap(partial(tokenize_randomneighbor_columns, config=self), cols, chunksize=100),total=len(cols)))
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

            with open("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_randomneighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"]
        }

###########################################

def tokenize_neighbor_columns(column_data, config):
    max_len_avail = config.max_length - (2 + config.window_size * 2)
    max_len_other_col = max_len_avail // (1 + config.window_size * 2)
    max_len_main_col = max_len_avail - (max_len_other_col * config.window_size * 2)
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

        if os.path.exists("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_neighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug)):
            print("Loading already processed {} dataset".format(split))

            with open("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_neighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), "rb") as f:
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
            self.split = split
            if (aug is not None) and (split != 'test') :
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None
            # self.augs = augs #.split('-')
            # so you would insert e.g. del-swap-summarize
            # don't support now

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
            processed_cols = list(tqdm(pool.imap(partial(tokenize_neighbor_columns, config=self), cols, chunksize=100),total=len(cols)))
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

            with open("../../data/CTA/{}/processed_datasets/cta_{}_{}_ml{}_win{}_preprocess{}_aug{}_neighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"]
        }

########       tabert        ########
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

def tokenize_tabert(column_data, config):
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
    
    type_func = lambda x:(x)
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

        if os.path.exists(f"../../data/CTA/{folder}/processed_datasets/cta_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_tabert.pkl"):
            print("Loading already processed {} dataset".format(split))

            with open(f"../../data/CTA/{folder}/processed_datasets/cta_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_tabert.pkl", "rb") as f:
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
            processed_cols = list(tqdm(pool.imap(partial(tokenize_tabert, config=self), cols, chunksize=100),total=len(cols)))
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

            with open(f"../../data/CTA/{folder}/processed_datasets/cta_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_tabert.pkl", 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"]
        }

############################################  

# CPA
# Austin
# Tabert    
def tokenize_column_pair_tabert(column_pair_data, config):
    # input: [table name, [main col index, col index], label_ids(list)]
    table = column_pair_data[0]
    main_column_index = column_pair_data[1][0]
    column_index = column_pair_data[1][1]
    label = column_pair_data[2]
    
    if config.split == 'train':
        path = f'../../data/CPA/{config.folder}/Train/' + table
    elif config.split == 'dev':
        path = f'../../data/CPA/{config.folder}/Validation/' + table
    else:
        path = f'../../data/CPA/{config.folder}/Test/' + table
    
    df_table = pd.read_json(path, compression='gzip', lines=True)

    # prepaeing main column data
    main_col_ml = int(config.max_length * config.main_percentage)
    other_col_ml = config.max_length - main_col_ml
    other_col_num = len(df_table.columns) - 1
    # other_col_average_ml = other_col_ml // other_col_num
    
    
    type_func = lambda x:map_types(x)
    
    tokenize_func = lambda x:config.tokenizer.encode_plus(text=x,
                                                  add_special_tokens=False,
                                                  padding=True,
                                                  truncation=True,
                                                  return_attention_mask=False,
                                                  max_length=other_col_ml).input_ids
    
    df_table_typed = df_table.astype("string").apply(np.vectorize(type_func))
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
    
class CPATaBertDataset(Dataset):

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

        # cta_cta_tabert_bert-base-uncased-bs-16_ml-128_seed-42_ws-0_aug-None.pt
        if os.path.exists(f"../../data/CPA/{folder}/processed_datasets/cpa_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_tabert.pkl"):
            print("Loading already processed {} dataset".format(split))
            
            with open(f"../../data/CPA/{folder}/processed_datasets/cpa_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_tabert.pkl", "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/CPA/{}/processed_datasets".format(folder)))
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
            processed_cols = list(tqdm(pool.imap(partial(tokenize_column_pair_tabert, config=self), column_pairs, chunksize=100),total=len(column_pairs)))
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

            with open(f"../../data/CPA/{folder}/processed_datasets/cpa_{split}_{bert}_ml{max_length}_win{window_size}_preprocess{preprocess}_aug{aug}_tabert.pkl", 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"]
        } 

# Reng 
# neighbor 
def cpa_tokenize_neighbor_columns(column_data, config):
    max_len_avail = config.max_length - (3 + config.window_size * 2)
    max_len_other_col = max_len_avail // (2 + config.window_size * 2) 
    max_len_target_col = max_len_avail - (max_len_other_col * (1 + config.window_size * 2))

    column_index = column_data[0]
    table_title = column_data[1]

    if config.split == 'train':
        path = '../../data/CPA/{}/Train/'.format(config.folder) + table_title
    elif config.split == 'dev':
        path = '../../data/CPA/{}/Validation/'.format(config.folder) + table_title
    else:
        path = '../../data/CPA/{}/Test/'.format(config.folder) + table_title

    df_table = pd.read_json(path, compression='gzip', lines=True)
     

    main_col_list = df_table[int('0')].astype("string").tolist()
    target_col_list = df_table[int(column_index)].astype("string").tolist()
    
    if config.aug != None:
        main_col_list = Augmenter.augment(main_col_list, config=config)
        target_col_list = Augmenter.augment(target_col_list, config=config)
        
    main_col_data = " ".join(main_col_list)
    target_col_data = " ".join(target_col_list) 
    
    main_col_token = config.tokenizer.encode_plus(main_col_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_other_col).input_ids
    target_col_token = config.tokenizer.encode_plus(target_col_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_target_col).input_ids

    input_ids = [config.tokenizer.cls_token_id] + main_col_token + [config.tokenizer.sep_token_id] + target_col_token + [config.tokenizer.sep_token_id]
    
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

class CPANeighborColumnDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 window_size: int = 0, #0/1/2
                 bert: str = 'bert' ,
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
        
        # _neighbor
        if os.path.exists("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_neighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug)):
            print("Loading already processed {} dataset".format(split))

            with open("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_neighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/CPA/{}/processed_datasets".format(folder)))
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
            self.split = split
            if (aug is not None) and (split != 'test') :
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None
            # self.augs = augs #.split('-')
            # so you would insert e.g. del-swap-summarize
            # don't support now

            cols_index = self.df["column_index"].tolist()
            cols_table_id = self.df["table_id"].tolist()

            cols = [list(x) for x in zip(cols_index, cols_table_id)]

#             if (self.aug == 'freq_cell_sampling'):
#                 if os.path.exists("../../data/CPA/{}/utils/common_cells.pkl".format(folder)):
#                     with open("../../data/CPA/{}/utils/common_cells.pkl".format(folder), "rb") as f:
#                         self.common_cells = pickle.load(f)[split]

#                     print("Loaded common cells")
#                 else:
#                     print("No common cells uploaded")
#                     raise SystemExit


            print('Processing {} {} columns'.format(len(cols), split))

            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(cpa_tokenize_neighbor_columns, config=self), cols, chunksize=100),total=len(cols)))
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

            with open("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_neighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"]
        }

### other random columns 
def cpa_tokenize_randomneighbor_columns(column_data, config):
    max_len_avail = config.max_length - (3 + config.window_size * 2)
    max_len_other_col = max_len_avail // (2 + config.window_size * 2) 
    max_len_target_col = max_len_avail - (max_len_other_col * (1 + config.window_size * 2))

    column_index = column_data[0]
    table_title = column_data[1]

    if config.split == 'train':
        path = '../../data/CPA/{}/Train/'.format(config.folder) + table_title
    elif config.split == 'dev':
        path = '../../data/CPA/{}/Validation/'.format(config.folder) + table_title
    else:
        path = '../../data/CPA/{}/Test/'.format(config.folder) + table_title

    df_table = pd.read_json(path, compression='gzip', lines=True)
     

    main_col_list = df_table[int('0')].astype("string").tolist()
    target_col_list = df_table[int(column_index)].astype("string").tolist()
    
    if config.aug != None:
        main_col_list = Augmenter.augment(main_col_list, config=config)
        target_col_list = Augmenter.augment(target_col_list, config=config)
        
    main_col_data = " ".join(main_col_list)
    target_col_data = " ".join(target_col_list) 
    
    main_col_token = config.tokenizer.encode_plus(main_col_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_other_col).input_ids
    target_col_token = config.tokenizer.encode_plus(target_col_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_target_col).input_ids

    input_ids = [config.tokenizer.cls_token_id] + main_col_token + [config.tokenizer.sep_token_id] + target_col_token + [config.tokenizer.sep_token_id]
    
    table_length = len(df_table.columns)
    remaining_index = list(filter(lambda x: x!= column_index, range(table_length)))
    num_to_pick = config.window_size * 2

    if len(remaining_index) < num_to_pick:
        index_to_use = (choices(remaining_index, k = num_to_pick))
    else:
        index_to_use = (random.sample(remaining_index, num_to_pick))

    for i in index_to_use:
        neighbor_data_list = df_table[int(i)].astype("string").tolist()
        if config.aug != None:
            neighbor_data_list = Augmenter.augment(neighbor_data_list, config=config)
        neighbor_data = " ".join(neighbor_data_list)

        neighbor_token = config.tokenizer.encode_plus(neighbor_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_other_col).input_ids

        input_ids = input_ids + neighbor_token + [config.tokenizer.sep_token_id]
        
    attention_mask = [1 for _ in range(len(input_ids))]
    assert(len(input_ids) == len(attention_mask))

    new_dict = {}
    for i in ('input_ids', 'attention_mask'):
        new_dict[i] = locals()[i]

    return new_dict

class CPARandomColumnDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 window_size: int = 0, #0/1/2
                 bert: str = 'bert' ,
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
        
        # _neighbor
        if os.path.exists("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_randomneighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug)):
            print("Loading already processed {} dataset".format(split))

            with open("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_randomneighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/CPA/{}/processed_datasets".format(folder)))
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
            self.split = split
            if (aug is not None) and (split != 'test') :
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None
            # self.augs = augs #.split('-')
            # so you would insert e.g. del-swap-summarize
            # don't support now

            cols_index = self.df["column_index"].tolist()
            cols_table_id = self.df["table_id"].tolist()

            cols = [list(x) for x in zip(cols_index, cols_table_id)]

#             if (self.aug == 'freq_cell_sampling'):
#                 if os.path.exists("../../data/CPA/{}/utils/common_cells.pkl".format(folder)):
#                     with open("../../data/CPA/{}/utils/common_cells.pkl".format(folder), "rb") as f:
#                         self.common_cells = pickle.load(f)[split]

#                     print("Loaded common cells")
#                 else:
#                     print("No common cells uploaded")
#                     raise SystemExit


            print('Processing {} {} columns'.format(len(cols), split))

            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(cpa_tokenize_randomneighbor_columns, config=self), cols, chunksize=100),total=len(cols)))
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

            with open("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_randomneighbor.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"]
        }
    
### summary serialization    
def detect_date(datestring): 
    try: 
        return pd.to_datetime(datestring, format='%Y-%m-%d', warn=False)
    except:
        return datestring
    
def get_summary_col(a_dict, column_index, max_len):
    target_var = a_dict.get('variables').get(column_index)
    keys = target_var.keys()
    if target_var.get('type') == 'DateTime': 
        target_tuple = ('n_distinct', 'n_unique', 'n_missing', 'count', 'type', 'min', 'max', 'range')
        target_tuple = tuple(filter(lambda x: x in keys, target_tuple))
        main_l = [[k, str(target_var[k])] for k in target_tuple]
        main_l = [" ".join(l) for l in main_l]
        values_l = [[k, str(v)] for k, v in target_var.get('value_counts_without_nan').items()]
        values_l = [" ".join(l) for l in values_l]
        final_str = " ".join(main_l + values_l)
    elif target_var.get('type') == 'Numeric':
        target_tuple = ('n_distinct','n_unique','n_missing','count','type','mean','std','variance','min','max','range','25%','50%','75%','iqr')
        target_tuple = tuple(filter(lambda x: x in keys, target_tuple))
        main_l = [[k, str(target_var[k])] for k in target_tuple]        
        main_l = [" ".join(l) for l in main_l]
        values_l = [[k, str(v)] for k, v in target_var.get('value_counts_without_nan').items()]
        values_l = [" ".join(l) for l in values_l]
        final_str = " ".join(main_l + values_l)
    elif target_var.get('type') == 'Boolean':
        target_tuple = ('n_distinct','n_unique','n_missing','count','type')
        target_tuple = tuple(filter(lambda x: x in keys, target_tuple))
        main_l = [[k, str(target_var[k])] for k in target_tuple]
        main_l = [" ".join(l) for l in main_l]
        values_l = [[k, str(v)] for k, v in target_var.get('value_counts_without_nan').items()]
        values_l = [" ".join(l) for l in values_l]
        final_str = " ".join(main_l + values_l)   
    # url has bug in the library
    elif target_var.get('type') == 'URL': 
        target_tuple = ('n_distinct','n_unique','n_missing','count','type','min_length','mean_length','median_length','max_length')
        target_tuple = tuple(filter(lambda x: x in keys, target_tuple))
        main_l = [[k, str(target_var[k])] for k in target_tuple]
        main_l = [" ".join(l) for l in main_l]
        values_l = [[k, str(v)] for k, v in target_var.get('value_counts_without_nan').items()]
        values_l = [" ".join(l) for l in values_l]
        alias_l = [[k, str(v)] for k, v in target_var.get('category_alias_counts').items()]
        alias_l = [" ".join(l) for l in alias_l]
        word_l = [[k, str(v)] for k, v in target_var.get('word_counts').items()]
        word_l = [" ".join(l) for l in word_l]
        scheme_l = [[k, str(v)] for k, v in target_var.get('scheme_counts').items()]
        scheme_l = [" ".join(l) for l in scheme_l]
        netloc_l = [[k, str(v)] for k, v in target_var.get('netloc_counts').items()]
        netloc_l = [" ".join(l) for l in netloc_l]
        final_str = " ".join(main_l + alias_l + scheme_l + netloc_l + values_l + word_l)
    elif target_var.get('type') == 'Categorical': 
        target_tuple = ('n_distinct','n_unique','n_missing','count','type','min_length','mean_length','median_length','max_length','n_category')
        target_tuple = tuple(filter(lambda x: x in keys, target_tuple))
        main_l = [[k, str(target_var[k])] for k in target_tuple]
        main_l = [" ".join(l) for l in main_l]
        values_l = [[k, str(v)] for k, v in target_var.get('value_counts_without_nan').items()]
        values_l = [" ".join(l) for l in values_l]
        alias_l = [[k, str(v)] for k, v in target_var.get('category_alias_counts').items()]
        alias_l = [" ".join(l) for l in alias_l]
        word_l = [[k, str(v)] for k, v in target_var.get('word_counts').items()]
        word_l = [" ".join(l) for l in word_l]
        final_str = " ".join(main_l + alias_l + values_l + word_l)
    else: # Unsupported 
        # or return the original string? 
        target_tuple = ('n_distinct','n_unique','n_missing','count','type')
        target_tuple = tuple(filter(lambda x: x in keys, target_tuple))
        main_l = [[k, str(target_var[k])] for k in target_tuple]
        main_l = [" ".join(l) for l in main_l]
        values_l = [[k, str(v)] for k, v in target_var.get('value_counts_without_nan').items()]
        values_l = [" ".join(l) for l in values_l]
        final_str = " ".join(main_l + values_l)
    alert_list = a_dict.get('alerts')
    alert_str = " ".join(alert_list)
    final_str = (final_str + ' ' + alert_str)
    return final_str

def cpa_tokenize_summary_columns(column_data, config):
    max_len_avail = config.max_length - 3
    max_len_main_col = max_len_avail // 2
    max_len_target_col = max_len_avail - max_len_main_col

    column_index = column_data[0]
    table_title = column_data[1]

    if config.split == 'train':
        path = '../../data/CPA/{}/Train/'.format(config.folder) + table_title
    elif config.split == 'dev':
        path = '../../data/CPA/{}/Validation/'.format(config.folder) + table_title
    else:
        path = '../../data/CPA/{}/Test/'.format(config.folder) + table_title

    df_table = pd.read_json(path, compression='gzip', lines=True)
    df_table.columns = df_table.columns.astype(str)

    sample_col = df_table[column_index].copy()
    sample_str = sample_col.loc[(sample_col.notnull()&sample_col.astype(bool)).idxmax()]

    if isinstance(detect_date(sample_str), datetime.date):
        df_table[column_index] = df_table[column_index].apply(detect_date)
    
    profile = ProfileReport(df_table, config_file="./profiling_minimal.yml", correlations=None)
    json_data = profile.to_json()
    a_dict = json.loads(json_data)
    
    main_col_data = get_summary_col(a_dict, "0", 256)
    target_col_data = get_summary_col(a_dict, str(column_index), 256)
    
    if config.aug != None:
        pass
        # main_col_list = Augmenter.augment(main_col_list, config=config)
        # target_col_list = Augmenter.augment(target_col_list, config=config)
    
    main_col_token = config.tokenizer.encode_plus(main_col_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_main_col).input_ids
    target_col_token = config.tokenizer.encode_plus(target_col_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_target_col).input_ids

    input_ids = [config.tokenizer.cls_token_id] + main_col_token + [config.tokenizer.sep_token_id] + target_col_token + [config.tokenizer.sep_token_id]

    attention_mask = [1 for _ in range(len(input_ids))]
    assert(len(input_ids) == len(attention_mask))

    new_dict = {}
    for i in ('input_ids', 'attention_mask'):
        new_dict[i] = locals()[i]

    return new_dict

class CPASummarizeColumnDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 window_size: int = 0, #0/1/2
                 bert: str = 'bert' ,
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

        if os.path.exists("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_summary.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug)):
            print("Loading already processed {} dataset".format(split))

            with open("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_summary.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/CPA/{}/processed_datasets".format(folder)))
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
            self.split = split
            if (aug is not None) and (split != 'test') :
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None
            # self.augs = augs #.split('-')
            # so you would insert e.g. del-swap-summarize
            # don't support now

            cols_index = self.df["column_index"].tolist()
            cols_table_id = self.df["table_id"].tolist()

            cols = [list(x) for x in zip(cols_index, cols_table_id)]

#             if (self.aug == 'freq_cell_sampling'):
#                 if os.path.exists("../../data/CPA/{}/utils/common_cells.pkl".format(folder)):
#                     with open("../../data/CPA/{}/utils/common_cells.pkl".format(folder), "rb") as f:
#                         self.common_cells = pickle.load(f)[split]

#                     print("Loaded common cells")
#                 else:
#                     print("No common cells uploaded")
#                     raise SystemExit

            print('Processing {} {} columns'.format(len(cols), split))
            test_dict = (cpa_tokenize_summary_columns(cols[1], config=self))

            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(cpa_tokenize_summary_columns, config=self), cols, chunksize=100),total=len(cols)))
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

            with open("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_summary.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"]
        }

######## end Reng ############    
######## Munir ############
# CPA cosine similarity
def clean_text(text):
    
    if(isinstance(text, dict)):
        text = ' '.join([ clean_text(v) for k, v in text.items()] )
    elif(isinstance(text, list)):
        text = map(clean_text, text)
        text = ' '.join(text)
        
    if pd.isnull(text):
        return ''
    
    #Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", str(text))
    re.sub(r'\d', '', text)

    #Remove excess whitespaces
    text = re.sub(' +', ' ', text).strip()
    return text


def convert_num2word(target_string):
    """ convert numbers like amount and date to words
        s.t. the tfidf method can be applied to those columns
    """
    is_list = False
    if not isinstance(target_string, str):
        target_string = str(target_string)
    # convert list to str, to use str.methods
    if isinstance(target_string, list):
        is_list = True
        target_string = ' | '.join(x for x in target_string)
    target_string2 = target_string
    for match in re.finditer(r'\d+', target_string):
        num = match.group(0)
        # avoid overflow error, just trim the number
        if len(num) > 32:
            num = num[:32]
        words = num2words(num)
        target_string2 = target_string2.replace(num, words)
    # return string back to list
    # maybe use another sep symbole to be sure, not to cut on wrong position
    if is_list:
        target_string = target_string.split(' | ')
        #print(target_string2)
    return target_string2

class TFIDF:
    def collection_vocab(self, df):
        out_of_vocabulary = set()
        vocab = defaultdict(lambda: [])
        df = df.astype("string").apply(np.vectorize(lambda x: convert_num2word(x)))
        for idx, col in enumerate(df.columns):
            #df_table_typed = df.astype("string").apply(np.vectorize(lambda x:map_types(x)))
            #t = Counter(df_table_typed[col]).most_common(1)[0][0]
            #if t in ['date', 'real_number']:
                #out_of_vocabulary.add(idx)
                #vocab[idx] = []
                #ontinue
            str_col = clean_text(" ".join([str(x) for x in df[col]]))
            stopwords_list = stopwords.words("english")
            stemmer = PorterStemmer()
            text = nltk.word_tokenize(str_col.lower())
            text = [
                    stemmer.stem(token.translate(str.maketrans('', '', string.punctuation))) 
                    for token in text if token not in stopwords_list and re.sub(r'\d', '', token) != '']
            text = list(filter(None, text))
            vocab[idx] = text
        return vocab, out_of_vocabulary
    
    def create_coll_dict(self, df):
        coll_vocabs = []# list of all vocab in table
        N = 0
        vocab, out_of_vocabulary = self.collection_vocab(df)
        for key, value in vocab.items():
            #if key not in out_of_vocabulary:
            N += 1
            coll_vocabs.extend(value)
            
        # need one safty check here, what if all columns are of number types
        coll_vocabs_dict = Counter(coll_vocabs)
        tf_idf = defaultdict(lambda: defaultdict(lambda: 0.0))
        for k, vals in vocab.items():
            counter = Counter(vals)
            if len(vals) == 0:
                # bi vicabm I shuould put continue here after this followin line
                tf_idf[k] = dict.fromkeys(tf_idf[k], 0)
                continue
            for val in np.unique(vals):
                tf = (1 + np.log10(counter[val])) / (1 + np.log10(counter.most_common(1)[0][1])) if counter[val] > 0 else 0
                df = 0
                # in how many docs the vocab appears
                for k2, v2 in vocab.items():
                    if val in v2:
                        df+=1
                try:
                    idf = np.log10(N/df) 
                    tf_idf[k][val] = tf * idf
                except:
                    tf_idf[k][val] = 0
                
        tf_idf = self._convert_to_df(tf_idf)
        #display(tf_idf)
        return tf_idf
    
    def _convert_to_df(self, tf_idf):
        return pd.DataFrame(tf_idf).fillna(0)
       
def cpa_tokenize_cosine_sim_columns(column_data, config):
    column_index = column_data[0]
    table_title = column_data[1]
    if config.split == 'train':
        path = '../../data/CPA/{}/Train/'.format(config.folder) + table_title
    elif config.split == 'dev':
        path = '../../data/CPA/{}/Validation/'.format(config.folder) + table_title
    else:
        path = '../../data/CPA/{}/Test/'.format(config.folder) + table_title

    df_table = pd.read_json(path, compression='gzip', lines=True)

    main_col_list = df_table[int('0')].astype("string").tolist()
    target_col_list = df_table[int(column_index)].astype("string").tolist()
    if config.aug != None:
        main_col_list = Augmenter.augment(main_col_list, config=config)
        target_col_list = Augmenter.augment(target_col_list, config=config)
        
    main_col_data = " ".join(main_col_list)
    target_col_data = " ".join(target_col_list) 
      
    df = df_table.astype('string')
    similarities = defaultdict(lambda: int)#np.zeros(df.shape[1])
    main_column_idx = int(column_index)
    tfidf = TFIDF()
    df_enc = tfidf.create_coll_dict(df)
    for col in df.columns:
        assert main_column_idx < df.shape[1], "Main column index is larger than the number of columns"
        if col == main_column_idx or col == 0:
            similarities[col] = 0 # assign 0
            continue # no need to compute similarity between column and itself
        else:
            x_enc = np.array(df_enc[main_column_idx])
            y_enc = np.array(df_enc[col])
            if 0 in [len(x_enc), len(y_enc)]:
                continue
            d = cosine_similarity(x_enc.reshape(1,-1), y_enc.reshape(1,-1)).tolist()
            #np.squeeze(d)
            while isinstance(d, list):
                d = d[0]
            similarities[col] = d
    #idx = np.argmax(similarities)

    #k = config.window_size * 2
    #idx = nlargest(config.window_size, similarities, key = similarities.get)
    idx = []
    for key, val in similarities.items():
        if val > 0.5:
            idx.append(key)
    len_idx = len(idx) if len(idx) > 0 else 1 # to avoid divide by zero, in case, nothing was greater thanthe threshold
    max_len_main_col = (config.max_length - (3 + len_idx))  // 2 
    max_len_target_col = config.max_length - max_len_main_col
    col_share = config.max_length // len_idx # each column will take part of max length
    extra_data= '' 
    for i in idx:
        col_list = df_table[int(i)].astype("string").tolist()
        if config.aug != None:
            extra_data_list = Augmenter.augment(col_list, config=config)
            extra_data += ' '.join(extra_data_list[:col_share])
        else:
            extra_data += ' '.join(col_list[:col_share])

    main_col_token = config.tokenizer.encode_plus(main_col_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=max_len_main_col).input_ids
    target_col_token = config.tokenizer.encode_plus(target_col_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=col_share).input_ids
    extra_data_token = config.tokenizer.encode_plus(extra_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=col_share).input_ids
    
    input_ids = [config.tokenizer.cls_token_id] + main_col_token + [config.tokenizer.sep_token_id] + target_col_token + [config.tokenizer.sep_token_id] + extra_data_token + [config.tokenizer.sep_token_id]
    attention_mask = [1 for _ in range(len(input_ids))]
    assert(len(input_ids) == len(attention_mask))
    #clear_output(wait=True)

    new_dict = {}
    for i in ('input_ids', 'attention_mask'):
        new_dict[i] = locals()[i]
    return new_dict

class CPACosineSimColumnDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 window_size: int = 0, #0/1/2
                 bert: str = 'bert' ,
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

        if os.path.exists("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_cosine_sim.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug)):
            print("Loading already processed {} dataset".format(split))

            with open("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_cosine_sim.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/CPA/{}/processed_datasets".format(folder)))
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
            self.split = split
            if (aug is not None) and (split != 'test') :
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None
            # self.augs = augs #.split('-')
            # so you would insert e.g. del-swap-summarize
            # don't support now

            cols_index = self.df["column_index"].tolist()
            cols_table_id = self.df["table_id"].tolist()

            cols = [list(x) for x in zip(cols_index, cols_table_id)]

            print('Processing {} {} columns'.format(len(cols), split))
            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(cpa_tokenize_cosine_sim_columns, config=self), cols, chunksize=100),total=len(cols)))
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

            with open("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_cosine_sim.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"]
        }

###### Munir End ##################
### Ichen
def remove_stopword(word_list):
    stop_words = set(stopwords.words('english'))
    clean_list = [word for word in word_list.split() if not word.lower() in stop_words]
    return clean_list
    
#[CLS] (text) first second third [SEP] main col [SEP] other col [SEP]
def tokenize_freq_three(column_pair_data, config):
    max_len = config.max_length - 7 # 4 tokenizers and 3 most frequent word 
    column_index = column_pair_data[0]
    table_title = column_pair_data[1]
    main_col_len = max_len // 2
    target_col_len = max_len - main_col_len

    if config.split == 'train':
        path = '../../data/CPA/{}/Train/'.format(config.folder) + table_title
    elif config.split == 'dev':
        path = '../../data/CPA/{}/Validation/'.format(config.folder) + table_title
    else:
        path = '../../data/CPA/{}/Test/'.format(config.folder) + table_title

    df_table = pd.read_json(path, compression='gzip', lines=True)

    main_col_list = df_table[int('0')].astype("string").tolist()
    target_col_list = df_table[int(column_index)].astype("string").tolist()
    
    if config.aug != None:
        main_col_list = Augmenter.augment(main_col_list, config=config)
        target_col_list = Augmenter.augment(target_col_list, config=config)
        
    main_col_data = " ".join(main_col_list)
    target_col_data = " ".join(target_col_list)
    #top_three_counts = Counter(remove_stopword(main_col_data)).most_common(3)
    top_three_counts = Counter(remove_stopword(target_col_data)).most_common(3)
    top_three_words = map(lambda x: x[0], top_three_counts)
    top_three_string = " ".join(top_three_words)
    
    most_freq_token = config.tokenizer.encode_plus(top_three_string, add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=3).input_ids
    main_col_token = config.tokenizer.encode_plus(main_col_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=main_col_len).input_ids
    target_col_token = config.tokenizer.encode_plus(target_col_data[:200000], add_special_tokens=False, padding=True, truncation=True, return_attention_mask=False, max_length=target_col_len).input_ids
    
    input_ids = [config.tokenizer.cls_token_id] + most_freq_token + [config.tokenizer.sep_token_id] + main_col_token + [config.tokenizer.sep_token_id] + target_col_token + [config.tokenizer.sep_token_id]

    attention_mask = [1 for _ in range(len(input_ids))]
    assert(len(input_ids) == len(attention_mask))

    new_dict = {}
    for i in ('input_ids', 'attention_mask'):
        new_dict[i] = locals()[i]

    return new_dict

class CPAFreqThreeDataset(Dataset):
    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 window_size: int = 0, #0/1/2
                 bert: str = 'bert' ,
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

        if os.path.exists("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_freqword.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug)):
            print("Loading already processed {} dataset".format(split))

            with open("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_freqword.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("../../data/CPA/{}/processed_datasets".format(folder)))
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
            if (aug is not None) and (split != 'test') :
                self.aug = aug
                self.augmenter = Augmenter()
            else:
                self.aug = None
                self.augmenter = None
                
            cols_index = self.df["column_index"].tolist()
            cols_table_id = self.df["table_id"].tolist()

            cols = [list(x) for x in zip(cols_index, cols_table_id)]

            print('Processing {} {} column pairs'.format(len(cols), split))
            
            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(tokenize_freq_three, config=self), cols, chunksize=100),total=len(cols)))
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

            with open("../../data/CPA/{}/processed_datasets/cpa_{}_{}_ml{}_win{}_preprocess{}_aug{}_freqword.pkl".format(folder, split, bert, max_length, window_size, preprocess, aug), 'wb') as f:
                pickle.dump(self.df, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"],
            "attention": self.df.iloc[idx]["attention_tensor"],
            "table_id": self.df.iloc[idx]["table_id"],
        }