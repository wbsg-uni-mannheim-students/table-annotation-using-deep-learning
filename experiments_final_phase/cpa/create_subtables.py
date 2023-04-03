import pickle 
import os 
import sys
import pandas as pd
import math

def split_dataframe(df, chunk_size = 10): 
    chunks = list()
    num_chunks = math.ceil(len(df) / chunk_size) 

    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    
    return chunks
    
if __name__ == "__main__":
    train_path = '../../data/CPA/NEW_FULL/Train/'
    filepath = '../../data/CPA/cpa_lm_no_data.pkl'
    with open(filepath, "rb") as f:
        ori_df_dict = pickle.load(f)
    ori_train_df = ori_df_dict.get('train')
    ori_dev_df = ori_df_dict.get('dev')
    ori_test_df = ori_df_dict.get('test')
    new_train = []

    if not os.path.exists('../../data/CPA/NEW_FULL/Train_Sub'):
        os.makedirs('../../data/CPA/NEW_FULL/Train_Sub')
        
    for idx, row in ori_train_df.iterrows():
        table_path = train_path + row.table_id
        df_table = pd.read_json(table_path, compression='gzip', lines=True)
    
        df_split = split_dataframe(df_table, 20)
        for i in range(min(len(df_split), 5)):
            tmp = df_split[i].reset_index(drop=True)
            new_table_id = row.table_id[:-8] + "_" + str(i) + ".json.gz"
            save_to = ("../../data/CPA/NEW_FULL/Train_Sub/"+ new_table_id)
            if os.path.exists(save_to):
                print(f"{save_to} exists")
                pass
            else:
                tmp.to_json(save_to, compression='gzip', orient='records', lines=True)
            new_train.append([new_table_id, row.labels, row.column_index, row.label_ids])
        
    new_train_df = pd.DataFrame(new_train, columns = ['table_id', 'labels', 'column_index', 'label_ids'])
    output_name = "../../data/CPA/cpa_lm_no_data_subtables.pkl"
    new_dict = {'train': new_train_df, 'dev': ori_dev_df, 'test': ori_test_df}
    f = open(output_name,'wb')
    pickle.dump(new_dict,f)
    f.close()