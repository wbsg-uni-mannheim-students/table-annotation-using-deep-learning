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
    main_column = column_pair_data[0]
    column = column_pair_data[1]
    label = column_pair_data[2]

    main_column_encoded = config.tokenizer.encode_plus(main_column[:200000], add_special_tokens=True, padding = True, truncation=True, return_attention_mask = True, max_length=config.max_length + 2)

    column_encoded = config.tokenizer.encode_plus(column[:200000], add_special_tokens=True, padding = True, truncation=True, return_attention_mask = True, max_length=config.max_length + 2)


    token_ids = main_column_encoded["input_ids"] + column_encoded["input_ids"]
    attention_masks = main_column_encoded["attention_mask"] + column_encoded["attention_mask"]

    return [token_ids, attention_masks, label]


class CPASingleColumnDataset(Dataset):
    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 bert: str = 'bert' ,
                 device: torch.device = None):


        if device is None:
            device = torch.device('cpu')

        if os.path.exists("data/CPA/processed_datasets/cpa_{}_{}_{}_singlecolumn.pkl".format(folder, split, bert, max_length)):
            print("Loading already processed {} dataset".format(split))
            with open("data/CPA/processed_datasets/cpa_{}_{}_{}_singlecolumn.pkl".format(folder, split, bert, max_length), "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict

        else:
            try:
                os.mkdir(os.path.join("data/CPA/processed_datasets"))
            except FileExistsError:
                pass

            #Open split
            with open(filepath, "rb") as f:
                df_dict = pickle.load(f)

            assert split in df_dict
            self.df = df_dict[split]
            self.tokenizer = tokenizer
            self.max_length = max_length

            #Tokenization of pairs of columns (main column + other column)
            column_pairs = [] #[main_column, other column, cpa label]

            #Main column dictionary
            main_columns = {}

            for index, row in self.df.loc[self.df["column_id"] == 0].iterrows():
                main_columns[row["table_id"]] = row["data"]

            for index, row in self.df.loc[self.df["column_id"] != 0].iterrows():
                column_pairs.append([main_columns[row["table_id"]], row["data"], row["label_ids"]])

            print('Processing {} {} column pairs'.format(len(column_pairs), split))

            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(tokenize_column_pair, config=self), column_pairs, chunksize=100),total=len(column_pairs)))
            pool.close()
            pool.join()

            input_ids = []
            attention_masks = []

            #Convert to tensors
            for processed_col in processed_cols:
                input_ids.append(torch.tensor(processed_col[0]))
                attention_masks.append(torch.tensor(processed_col[1]))

            #Remove main columns
            self.df = self.df.loc[self.df["column_id"] != 0]

            self.df["data_tensor"] = input_ids
            self.df["attention_tensor"] = attention_masks
            self.df["label_tensor"] = self.df["label_ids"].apply(lambda x: torch.tensor([x]))

            with open("data/CPA/processed_datasets/cpa_{}_{}_{}_singlecolumn.pkl".format(folder, split, bert, max_length), 'wb') as f:
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
def map_types(cell_value):
    match_dict = {"url" : r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])",
                  "date" : r"\d{4}-[0,1]{0,1}[0-9]{0,1}-[0-3]{0,1}[0-9]{0,1}",
                  "real_number" : r"(\d+\.?\d*)"

    }
    if cell_value == " " or cell_value == "" or cell_value == "N/A" :
        return "null"
    if re.match(match_dict["url"], cell_value) is not None:
        if re.match(match_dict["url"], cell_value).group(0) == cell_value:
            return "url"
        else:
            return "text"

    elif re.match(match_dict["date"], cell_value) is not None:
        if re.match(match_dict["date"], cell_value).group(0) == cell_value:
            return "date"
        else:
            return "text"
    elif re.match(match_dict["real_number"], cell_value) is not None:
        if re.match(match_dict["real_number"], cell_value).group(0) == cell_value:
            return "real_number"
        else:
            return "text"
    else:
        return "text"

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

    main_col_ml = int(config.max_length * config.main_percentage)
    other_col_ml = config.max_length - main_col_ml
    other_col_num = len(df_table.columns) - 1
    # other_col_average_ml = other_col_ml // other_col_num
    excepted_row_num = 2 * (other_col_ml//other_col_num) # at least 1 tokens: datatype

    type_func = lambda x:map_types(x) +" "+ x
#     # mean version
#     tokenize_func = lambda x:config.tokenizer.encode_plus(text=x,
#                                                   add_special_tokens=False,
#                                                   padding=True,
#                                                   truncation=True,
#                                                   return_attention_mask=False,
#                                                   max_length=other_col_average_ml).input_ids

    tokenize_func = lambda x:config.tokenizer.encode_plus(text=x,
                                                  add_special_tokens=False,
                                                  padding=True,
                                                  truncation=True,
                                                  return_attention_mask=False,
                                                  max_length=other_col_ml).input_ids

    df_table_typed = df_table.astype("string").apply(np.vectorize(type_func))
    main_col_list = df_table_typed[int(column_index)].astype("string").tolist()
    if config.aug != None:
        main_col_list = Augmenter.augment(main_col_list, config=config)
    main_col = " ".join(main_col_list)
    main_col_tokens = config.tokenizer.encode_plus(text=main_col,
                                                  add_special_tokens=False,
                                                  padding=True,
                                                  truncation=True,
                                                  return_attention_mask=False,
                                                  max_length=main_col_ml).input_ids

    other_col_table =  df_table_typed.drop(int(column_index), axis=1)

    other_tokens = []
    if excepted_row_num >= df_table_typed.shape[0]:
        excepted_row_num = df_table_typed.shape[0]
    for row in range(excepted_row_num):
        other_tokens += [config.tokenizer.sep_token_id]
        row_data = " ".join(other_col_table.loc[row,:].tolist())
        other_tokens += tokenize_func(row_data)
#         other_tokens += other_col_table.loc[row,:].apply(tokenize_func).sum()
    other_tokens = other_tokens[:other_col_ml]

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
