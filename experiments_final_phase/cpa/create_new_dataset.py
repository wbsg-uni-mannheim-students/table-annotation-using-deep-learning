import pandas as pd
import os
import re
import multiprocessing
import numpy as np
import tqdm
from scipy import stats
from functools import partial

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

    #Remove excess whitespaces
    text = re.sub(' +', ' ', str(text)).strip()

    return text

def clean_text_tuta(text):
    text = clean_text(text)

    #Fixed each cell length to 8
    text = ' '.join(text.split(" ")[:8])

    return text

def get_cell_length(col):
    # get each cell length for the column
    cell_length = col.apply(str).apply(str.split).apply(len)
    # filter out cell with len = 0
    cell_length = cell_length.loc[lambda x: x > 0] if cell_length.loc[lambda x: x > 0].any() else 0
    return cell_length

def clean_text_length(text, length):
    text = clean_text(text)

    #Fixed each cell length to specified length of each column
    text = ' '.join(text.split()[:length])

    return text

def get_table(file_name):
    status = ""
    if file_name in cta_train_gt['table_name'].tolist():
        status = "/Train/"
        path = '../../data/CTA/Train/' + file_name
    elif file_name in cta_val_gt['table_name'].tolist():
        status = "/Validation/"
        path = '../../data/CTA/Validation/' + file_name
    else:
        status = "/Test/"
        path = '../../data/CTA/Test/' + file_name
    save_to = save_path + status + file_name
    df = pd.read_json(path, compression='gzip', lines=True)
    df = df.apply(np.vectorize(clean_text))

    ## for tuta
    # df = df.apply(np.vectorize(clean_text_tuta))

    ## for median
    #for col in df.columns:
    #    median = int(np.median(get_cell_length(df[col])))
    #    df[col] = df[col].apply(clean_text_length, length = median)
    
    ## for mean
    #for col in df.columns:
    #    mean = int(np.mean(get_cell_length(df[col])))
    #    df[col] = df[col].apply(clean_text_length, length = mean)
    
    if os.path.exists(save_to):
        print(f"{file_name} exists")
        pass
    else:
        df.to_json(save_to, compression='gzip', orient='records', lines=True)
    return 1

if __name__ == "__main__":
    cta_train_gt = pd.read_csv('../../data/CTA/CTA_training_gt.csv')
    cta_val_gt = pd.read_csv('../../data/CTA/CTA_validation_gt.csv')
    cta_test_gt = pd.read_csv('../../data/CTA/CTA_test_gt.csv')

    
    ## For no preprocess - NEW_FULL
    ## change for tuta/median/mean - TUTA_FULL etc
    save_path = "../../data/CTA/NEW_FULL"

    train_table_list = cta_train_gt.loc[:,"table_name"].drop_duplicates().tolist()
    val_table_list = cta_val_gt.loc[:,"table_name"].drop_duplicates().tolist()
    test_table_list = cta_test_gt.loc[:,"table_name"].drop_duplicates().tolist()

    pool = multiprocessing.Pool(processes=10)
    train_result = list(tqdm.tqdm(pool.imap(get_table, train_table_list), total=len(train_table_list)))
    val_result = list(tqdm.tqdm(pool.imap(get_table, val_table_list), total=len(val_table_list)))
    test_result = list(tqdm.tqdm(pool.imap(get_table, test_table_list), total=len(test_table_list)))
    pool.close()
    pool.join()
