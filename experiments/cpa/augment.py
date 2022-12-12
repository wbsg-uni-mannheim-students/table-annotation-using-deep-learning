import json
import random
import numpy as np
import pandas as pd
import nlpaug.augmenter.word as naw
import collections

class Config(object):
    pass

# https://stackoverflow.com/questions/71642183/python-structuring-a-project-with-utility-functions-shared-across-modules-at-di

# reng - shuffle_col_cell_val
def shuffle_cell(cell):
    word_list = cell.split(" ")
    return " ".join(random.sample(word_list, len(word_list)))

# reng - freq_cell_sampling
def success(succ_range): 
    return succ_range[0] <= random.uniform(0, 1) <= succ_range[1]

def apply_change(cell_val, common_cells):
    succ_range = [0.05, 0.2] 
    if success(succ_range): 
        to_replace = random.choice(common_cells)
        return to_replace
    return cell_val

# munir
def swap_row_cells(row):
    indecies = np.arange(len(list(row)))
    r = np.random.permutation(indecies)
    row = pd.Series([row[idx] for idx in r])
    return row

# ichen
def replace_word(column):
    nlp_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.1)
    augmented_list = [''.join(nlp_aug.augment(cell)) for cell in column]
    return augmented_list

# ichen
def swap_word(column):
    nlp_aug = naw.RandomWordAug(action = 'swap', aug_p=0.1)
    augmented_list = [''.join(nlp_aug.augment(cell)) for cell in column]
    return augmented_list

# reng
def back_translate(column): 
    nlp_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de', 
        to_model_name='facebook/wmt19-de-en'
    )
    # does not handle weird cases 
    augmented_list = [''.join(nlp_aug.augment(cell)) for cell in column]
    return augmented_list 

# munir
def replace_nan_with_freq_value(col_list):
    """replace Nan values with most frequent word in col
     to avoid expensive loops, if the counter method did not 
     work, in case of list of lists as input, the method will choose
     the first not Nan value it faces
    """
    if len(col_list) > 0:
        if (None in col_list) or ('' in col_list):
            try:
                counts = collections.Counter(col_list)
                most_common = counts.most_common(1)[0][0]
            except:
                # counter will throw an exception if col_list is a list of lists
                # instead take the first no none value
                most_common = next(item for item in col_list if ((item is not None)&(item != '')))
                if isinstance(most_common, list) and len(most_common) > 0:
                    most_common = most_common[0] 
            if most_common is None:
                most_common = next(item for item in col_list if ((item is not None)&(item != '')))
                if isinstance(most_common, list) and len(most_common) > 0:
                    most_common = most_common[0]    
        col_list = [item if ((item is not None)&(item != '')) else most_common for item in col_list]            
    return col_list
### 
    
class Augmenter(object):
    """Data augmentation operator.
    """
    def __init__(self):
        pass

    def augment(data_list, config):
        # it is done on a list 
        # but you can run this as a script to the 
        op = config.aug
        # reng 
        if op == 'shuffle_col_cell':
            data_list = data_list.copy() 
            data_list = random.sample(data_list, len(data_list))  
            return data_list
        elif op == 'shuffle_col_cell_val': 
            data_list = data_list.copy()
            shuffled_data_list = list(map(shuffle_cell, data_list))
            return shuffled_data_list
        elif op == 'back_translation': 
            data_list = data_list.copy() 
            translated_data_list = back_translate(data_list)
            return translated_data_list
        # not assigned 
        elif op == 'swap_row_cell':
            # todo - what did you include from the row, i.e. attribute + value
            pass
        # munir - 
        elif op == 'delete_random_cell':
            indices = np.random.choice(len(data_list), (len(data_list) // 2))# 2 can be a hayperparameter
            column_data_list = [row for idx, row in enumerate(data_list) if idx not in indices]
            return column_data_list
        # ichen 
        elif op == 'replace_word':
            data_list = data_list.copy()
            replace_list = replace_word(data_list)
            return replace_list
        # ichen
        elif op == 'swap_word':
            data_list = data_list.copy()
            swap_list = swap_word(data_list)
            return swap_list
        # munir - need to change to also handle row
        elif op == "replace_non_freq":
            freq_list = replace_nan_with_freq_value(data_list)
            return freq_list
        else: 
            return data_list 

if __name__ == '__main__':
    ag = Augmenter()