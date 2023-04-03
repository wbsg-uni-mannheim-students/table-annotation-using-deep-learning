#Doduo code from their github repository: https://github.com/megagonlabs/doduo
import random

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import torch


def f1_score_multilabel(true_list, pred_list):
    conf_mat = multilabel_confusion_matrix(np.array(true_list),
                                           np.array(pred_list))
    agg_conf_mat = conf_mat.sum(axis=0)
    # Note: Pos F1
    # [[TN FP], [FN, TP]] if we consider 1 as the positive class
    p = agg_conf_mat[1, 1] / agg_conf_mat[1, :].sum()
    r = agg_conf_mat[1, 1] / agg_conf_mat[:, 1].sum()
    
    micro_f1 = 2 * p * r / (p  + r) if (p + r) > 0 else 0.
    class_p = conf_mat[:, 1, 1] /  conf_mat[:, 1, :].sum(axis=1)
    class_r = conf_mat[:, 1, 1] /  conf_mat[:, :, 1].sum(axis=1)
    class_f1 = np.divide(2 * (class_p * class_r), class_p + class_r,
                         out=np.zeros_like(class_p), where=(class_p + class_r) != 0)
    class_f1 = np.nan_to_num(class_f1)
    macro_f1 = class_f1.mean()
    return (micro_f1, macro_f1, class_f1, conf_mat)

def set_seed(seed: int):
    """https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L58-L63"""    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """Add the following 2 lines
    https://discuss.pytorch.org/t/how-could-i-fix-the-random-seed-absolutely/45515
    """
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    """
    For detailed discussion on the reproduciability on multiple GPU
    https://discuss.pytorch.org/t/reproducibility-over-multigpus-is-impossible-until-randomness-of-threads-is-controled-and-yet/47079
    """
    
def replace_special_tokens(text):
    r = {
        'type': '[T]',
        'median_length': '[MEL]',
        'max_length': '[MAL]',
        'min_length': '[MIL]',
        'n_distinct': '[D]',
        'min': '[MI]',
        'max': '[MA]',
        'mean': '[ME]',
        'std': '[S]',
    }
    
    for key in r:
        text = text.replace(key, r[key])
    return text