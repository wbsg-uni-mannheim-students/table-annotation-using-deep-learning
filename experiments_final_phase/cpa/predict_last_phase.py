##Code adapted from DODUO: https://github.com/megagonlabs/doduo
import argparse
import json
import math
import random
from time import time
import os

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset_last_phase import (
    collate_fn,
    #CPA
    CPASumNeighborColumnDataset,
    CPAColTaBertDataset, 
    CPANoMainNeighborColumnDataset, 
    CPATaBertDataset, 
    CPANeighborColumnDataset, 
    CPASingleColumnDataset
)

import sys
sys.path.insert(0,'../..')

from model import BertForMultiOutputClassification, BertMultiPairPooler
from util import f1_score_multilabel, set_seed, replace_special_tokens

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",
        type=str,
        help="Huggingface model shortcut name ",
    )
    parser.add_argument(
        "--max_length",
        default=32,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--window_size",
        default=0,
        type=int,
        help="Number of column pairs, 1 or 2 pairs",
    )
    parser.add_argument(
        "--aug",
        type=str,
        help="Augmentation to do, None if not specified",
    )
    parser.add_argument(
        "--epoch",
        default=5,
        type=int,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Random seed",
    )
    parser.add_argument("--warmup",
                        type=float,
                        default=0.,
                        help="Warmup ratio")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--mp", type=float, default=0.5, help="Main percentage")
    # TODO
    parser.add_argument("--method",
                        type=str,
                        nargs="+",
                        default=["cta"],
                        choices=[
                            "cta", "cpa", "doduo"
                        ],
                        help="Task names}")
    # TODO
    parser.add_argument("--serialization",
                    type=str,
                    nargs="+",
                    default="single-column",
                    choices=[
                        "nomain_neighbor", "sum_neighbor", "col_tabert", "tabert", "single_col", "neighbor"
                    ],
                    help="Serialization methods}")
    parser.add_argument("--preprocess",
                            type=str,
                            help="Preprocessing to do, only support tuta, median and mean now",
                        )
    args = parser.parse_args()

    method = args.method[0]
    serialization = args.serialization[0]

    if method == 'cta':
        tasks = ['cta']
    elif method == 'cpa':
        tasks = ['cpa']
    else:
        tasks = ['cta', 'cpa']

    if method == 'doduo':
        assert serialization == 'all-table' , "Doduo (multi-task method) can be used only with all-table serialization"

    if 'roberta' in args.model_name:
        assert serialization != 'all-table', 'Roberta models can not be used with all table serialization (needs to be added)'

    task_num_class_dict = {
            "cta": 91,
            "cpa": 176,
        }

    filepaths_task_dict = {
        "cta": "../../data/CTA/NEW_FULL/cta_lm_no_data.pkl",
        "cpa": "../../data/CPA/NEW_FULL/cpa_lm_no_data.pkl",
    }
# TODO add another serialization
    serialization_method_dict = {
        "cpa": {
            "col_tabert": CPAColTaBertDataset, 
            "sum_neighbor": CPASumNeighborColumnDataset,
            "nomain_neighbor": CPANoMainNeighborColumnDataset,
            "tabert": CPATaBertDataset, 
            "neighbor": CPANeighborColumnDataset,
            "single_col": CPASingleColumnDataset
        }
    }

    if not os.path.exists('model/'):
        print("{} not exists. Created".format('model/'))
        os.makedirs("model/")

    #Tokenizer based on language model
    if args.serialization == "sum_neighbor":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, additional_special_tokens=('[T]', '[MEL]', '[MAL]', '[MIL]', '[D]', '[MI]', '[MA]', '[ME]', '[S]', '[VAL]'))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        
    if 'roberta' in args.model_name:
        base_model = 'roberta'
    else:
        base_model = 'bert'
        
    for task in tasks:

        model_path = "model/{}_{}_{}_{}-lr-{}_bs-{}_ml-{}_seed-{}_ws-{}_preprocess-{}_aug-{}_mp-{}.pt".format(method, task, serialization, args.model_name, args.lr, args.batch_size, args.max_length, args.random_seed, args.window_size, args.preprocess, args.aug, args.mp)
        print(model_path)

        # todo
        if (serialization == 'col_tabert') or (serialization == 'sum_neighbor') or (serialization == 'nomain_neighbor') or (serialization == 'tabert')  or (serialization == 'neighbor') or (serialization == 'single_col'):
            #Choose model
            if 'roberta' in args.model_name:
                model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=task_num_class_dict[task])
                model = model.to(device)         
            else:
                model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=task_num_class_dict[task])
                model = model.to(device)         

            #Choose serialization
            dataset_serialization = serialization_method_dict[task][serialization]

        #Add more conditions when adding new serialization methods
        else:
            if 'roberta' in args.model_name:
                pass
                # model = RobertaForMultiOutputClassification.from_pretrained(
                #         args.model_name,
                #         num_labels=task_num_class_dict[task],
                #         output_attentions=False,
                #         output_hidden_states=False,
                #     ).to(device)
            else:
                model = BertForMultiOutputClassification.from_pretrained(
                        args.model_name,
                        num_labels=task_num_class_dict[task],
                        output_attentions=False,
                        output_hidden_states=False,
                    ).to(device)

            dataset_serialization = serialization_method_dict[task][serialization]

            #What is the difference: using multipair pooler instead of usual pooler
            if task == "cpa":
                print("Use column-pair pooling")
                #Change for Roberta!!!
                # Use column pair embeddings
                config = BertConfig.from_pretrained(args.model_name)
                model.bert.pooler = BertMultiPairPooler(config).to(device)

        #Load test datasets and datasetloaders
        test_dataset = dataset_serialization(filepath=filepaths_task_dict[task],
                                    split="test",
                                    tokenizer=tokenizer,
                                    window_size=args.window_size,
                                    aug=args.aug,
                                    preprocess=args.preprocess,
                                    max_length=args.max_length,
                                    main_percentage=args.mp,
                                    bert=base_model,
                                    device=device)

        test_dataloader = DataLoader(test_dataset,
                                    batch_size=args.batch_size,
                                    collate_fn=collate_fn)
        
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_predictions = []
        test_labels = []

        eval_dict = {}
        for batch_idx, batch in enumerate(test_dataloader):
            # todo 
            if (serialization == 'col_tabert') or (serialization == 'sum_neighbor') or (serialization == 'nomain_neighbor') or (serialization == 'tabert') or (serialization == 'neighbor') or (serialization == 'single_col'):

                batch_input_ids = batch["data"].T.to(device)
                batch_mask = batch["attention"].T.to(device)
                #For cross-entropy loss labels should not be vectors
                batch_labels = torch.tensor([label.tolist().index(1) for label in batch["label"]]).to(device)

                loss, logits = model(batch_input_ids, token_type_ids=None, attention_mask=batch_mask, labels=batch_labels, return_dict=False)

                for p in logits.argmax(axis=-1):
                    y = [0] * logits.shape[1]
                    y[p] = 1
                    test_predictions.append(y)

                test_labels += batch["label"].cpu().detach().numpy().tolist()
            else:
                logits, = model(input_ids = batch["data"].T)

                # Align the tensor shape when the size is 1
                if len(logits.shape) == 2:
                    logits = logits.unsqueeze(0)

                cls_indexes = torch.nonzero( batch["data"].T == tokenizer.cls_token_id)
                filtered_logits = torch.zeros(cls_indexes.shape[0], logits.shape[2]).to(device)

                #Mark where CLS tokens are located
                for n in range(cls_indexes.shape[0]):
                    i, j = cls_indexes[n]
                    logit_n = logits[i, j, :]
                    filtered_logits[n] = logit_n

                if task == 'cta':
                    for pred in filtered_logits.argmax(axis=-1):
                        y = [0] * filtered_logits.shape[1]
                        y[pred] = 1
                        test_predictions.append(y)

                    test_labels += batch["label"].cpu().detach().numpy().tolist()

                else:
                    all_preds = []
                    for pred in filtered_logits.argmax(axis=-1):
                        y = [0] * filtered_logits.shape[1]
                        y[pred] = 1
                        all_preds.append(y)

                    all_labels = batch["label"].cpu().detach().numpy()
                    # Ignore the very first CLS token
                    idxes = np.where(all_labels > 0)[0]
                    test_predictions += [ pred for i, pred in enumerate(all_preds) if i in idxes ]
                    test_labels += [label.tolist() for label in batch["label"] if 1 in label.tolist()]

        ts_micro_f1, ts_macro_f1, ts_class_f1, ts_conf_mat = f1_score_multilabel(test_labels, test_predictions)

        eval_dict["ts_micro_f1"] = ts_micro_f1
        if type(ts_class_f1) != list:
            ts_class_f1 = ts_class_f1.tolist()
        eval_dict["ts_class_f1"] = ts_class_f1
        if type(ts_conf_mat) != list:
            ts_conf_mat = ts_conf_mat.tolist()
        eval_dict["confusion_matrix"] = ts_conf_mat

        print("test_macro_f1={:.4f} test_micro_f1={:.4f} "
            .format(ts_macro_f1, ts_micro_f1))

        if not os.path.exists('test_reports_final/'):
            print("{} not exists. Created".format('test_reports_final/'))
            os.makedirs("test_reports_final/")
            
        output_filepath = "test_reports_final/{}_{}_{}_{}-lr-{}_bs-{}_ml-{}_seed-{}_ws-{}_preprocess-{}_aug-{}_mp-{}.json".format(method, task, serialization, args.model_name, args.lr, args.batch_size, args.max_length, args.random_seed, args.window_size, args.preprocess, args.aug, args.mp)
        with open(output_filepath, "w") as fout:
            json.dump(eval_dict, fout)