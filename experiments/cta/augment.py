import json
import random
import numpy as np
import pandas as pd
import nlpaug.augmenter.word as naw

# https://stackoverflow.com/questions/71642183/python-structuring-a-project-with-utility-functions-shared-across-modules-at-di

def shuffle_cell(cell):
    word_list = cell.split(" ")
    return " ".join(random.sample(word_list, len(word_list)))

def success(succ_range):
    return succ_range[0] <= random.uniform(0, 1) <= succ_range[1]

def apply_change(cell_val, common_cells):
    succ_range = [0.05, 0.2]
    if success(succ_range):
        to_replace = random.choice(common_cells)
        return to_replace
    return cell_val

def swap_row_cells(row):
    indecies = np.arange(len(list(row)))
    r = np.random.permutation(indecies)
    row = pd.Series([row[idx] for idx in r])
    return row

def replace_word(column):
    naw.SynonymAug()
    augmented_list = [''.join(aug.augment(cell)) for cell in column]
    return augmented_list

def swap_word(column):
    aug = naw.RandomWordAug(action = 'swap')
    augmented_list = [''.join(aug.augment(cell)) for cell in column]
    return augmented_list

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
        if op == 'swap_col_cell':
            data_list = data_list.copy()
            data_list = random.sample(data_list, len(data_list))
            return data_list
        elif op == 'shuffle_col_val':
            data_list = data_list.copy()
            shuffled_data_list = list(map(shuffle_cell, data_list))
            return shuffled_data_list
        elif op == 'freq_cell_sampling':
            common_cells = config.common_cells
            data_list = data_list.copy()
            sampled = [apply_change(item, common_cells) for item in data_list]
            return sampled
        # munir
        elif op == 'swap_row_cell':
            # todo - what did you include from the row, i.e. attribute + value
            pass
        # munir
        elif op == 'delete_random_rows':
            indecies = np.random.choice(len(data_list), (len(data_list) // 2))# 2 can be a hayperparameter
            column_data_list = [row for idx, row in enumerate(data_list) if idx not in indecies]
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
        else:
            return data_list

    # munir
    # can also add more params for things you need
    def augment_table(df, config):
        op = config.aug
        op = config.aug
        df = df.apply(swap_row_cells, axis=1)
        return df

if __name__ == '__main__':
    ag = Augmenter()
