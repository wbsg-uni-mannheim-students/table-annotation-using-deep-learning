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
    
    return config.tokenizer.encode_plus(column_data[:200000], add_special_tokens=True, padding = True, truncation=True, return_attention_mask = True, max_length=config.max_length + 2)

class CTASingleColumnDataset(Dataset):
    
    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 bert: str = 'bert' ,
                 device: torch.device = None):
        
        if device is None:
            device = torch.device('cpu')
                
        if os.path.exists("data/CTA/processed_datasets/cta_{}_{}_{}_singlecolumn.pkl".format(split, bert, max_length)):
            print("Loading already processed {} dataset".format(split))
            with open("data/CTA/processed_datasets/cta_{}_{}_{}_singlecolumn.pkl".format(split, bert, max_length), "rb") as f:
                df_dict = pickle.load(f)
            self.df = df_dict
            
        else:
            try:
                os.mkdir(os.path.join("data/CTA/processed_datasets"))
            except FileExistsError:
                pass
            
            #Open split
            with open(filepath, "rb") as f:
                df_dict = pickle.load(f)

            assert split in df_dict
            self.df = df_dict[split]
            self.tokenizer = tokenizer
            self.max_length = max_length

            #Tokenization
            cols = self.df["data"].tolist()
            print('Processing {} {} columns'.format(len(cols), split))

            pool = Pool(processes=10)
            processed_cols = list(tqdm(pool.imap(partial(tokenize_columns, config=self), cols, chunksize=100),total=len(cols)))
            pool.close()
            
            input_ids = []
            attention_masks = []
            
            for processed_col in processed_cols:
                input_ids.append(torch.tensor(processed_col["input_ids"]))
                attention_masks.append(torch.tensor(processed_col["attention_mask"]))
            
            self.df["data_tensor"] = input_ids
            self.df["attention_tensor"] = attention_masks
            self.df["label_tensor"] = self.df["label_ids"].apply(lambda x: torch.tensor([x]))
            
            with open("data/CTA/processed_datasets/cta_{}_{}_{}_singlecolumn.pkl".format(split, bert, max_length), 'wb') as f:
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
            
        if os.path.exists("data/CTA/processed_datasets/cta_{}_{}_{}_alltable.pkl".format(split, bert, max_length)):
            print("Loading already processed {} dataset".format(split))
            with open("data/CTA/processed_datasets/cta_{}_{}_{}_alltable.pkl".format(split, bert, max_length), "rb") as f:
                df_dict = pickle.load(f)
            self.table_df = df_dict
            
        else:
            try:
                os.mkdir(os.path.join("data/CTA/processed_datasets"))
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
            
            with open("data/CTA/processed_datasets/cta_{}_{}_{}_alltable.pkl".format(split, bert, max_length), 'wb') as f:
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
                
        if os.path.exists("data/CPA/processed_datasets/cpa_{}_{}_{}_singlecolumn.pkl".format(split, bert, max_length)):
            print("Loading already processed {} dataset".format(split))
            with open("data/CPA/processed_datasets/cpa_{}_{}_{}_singlecolumn.pkl".format(split, bert, max_length), "rb") as f:
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
            
            with open("data/CPA/processed_datasets/cpa_{}_{}_{}_singlecolumn.pkl".format(split, bert, max_length), 'wb') as f:
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
            
        if os.path.exists("data/CPA/processed_datasets/cpa_{}_{}_{}_alltable.pkl".format(split, bert, max_length)):
            print("Loading already processed {} dataset".format(split))
            with open("data/CPA/processed_datasets/cpa_{}_{}_{}_alltable.pkl".format(split, bert, max_length), "rb") as f:
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
            
            with open("data/CPA/processed_datasets/cpa_{}_{}_{}_alltable.pkl".format(split, bert, max_length), 'wb') as f:
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
