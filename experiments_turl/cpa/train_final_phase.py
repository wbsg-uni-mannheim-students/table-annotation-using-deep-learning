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
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset_final_phase import (
    collate_fn,
    #CTA
    CTAColTaBertDataset,
    #CPA
    CPAColTaBertDataset, 
    CPATaBertDataset, 
    CPANoMainNeighborColumnDataset
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
                        "nomain_neighbor", "sum_neighbor", "col_tabert", "tabert"
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
            "cta": 255,
            "cpa": 121,
        }

    filepaths_task_dict = {
        "cta": "../../data/turl_cta_data/cta_turl_lm.pkl",
        "cpa": "../../data/turl_cpa_data/cpa_turl_lm.pkl",
    }
# TODO add another serialization
    serialization_method_dict = {
        "cta": {
            "col_tabert": CTAColTaBertDataset, 
        },
        "cpa": {
            "col_tabert": CPAColTaBertDataset, 
            "tabert": CPATaBertDataset,
            "nomain_neighbor": CPANoMainNeighborColumnDataset
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

    #If Doduo there are two models: one for CTA and one for CPA
    models = []
    train_datasets = []
    train_dataloaders = []
    valid_datasets = []
    valid_dataloaders = []

    for task in tasks:
        # TODO add serialization
        if (serialization == 'col_tabert') or (serialization == 'sum_neighbor') or (serialization == 'nomain_neighbor') or (serialization == 'tabert'):
            #Choose model
            if 'roberta' in args.model_name:
                model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=task_num_class_dict[task])
                model.resize_token_embeddings(len(tokenizer))
            else:
                model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=task_num_class_dict[task])
                model.resize_token_embeddings(len(tokenizer))
            dataset_serialization = serialization_method_dict[task][serialization]
        else:
            pass

        print("Task is {}_{}_{}_{}-lr-{}_bs-{}_ml-{}_seed-{}_ws-{}_preprocess-{}_aug-{}_mp-{}".format(method, task, serialization, args.model_name, args.lr, args.batch_size, args.max_length, args.random_seed, args.window_size, args.preprocess, args.aug, args.mp))

        train_dataset = dataset_serialization(filepath=filepaths_task_dict[task],
                                        split="train",
                                        tokenizer=tokenizer,
                                        window_size=args.window_size,
                                        aug=args.aug,
                                        preprocess=args.preprocess,
                                        max_length=args.max_length,
                                        main_percentage=args.mp,
                                        bert=base_model,
                                        device=device)

        valid_dataset = dataset_serialization(filepath=filepaths_task_dict[task],
                                    split="dev",
                                    tokenizer=tokenizer,
                                    window_size=args.window_size,
                                    aug=args.aug,
                                    preprocess=args.preprocess,
                                    max_length=args.max_length,
                                    main_percentage=args.mp,
                                    bert=base_model,
                                    device=device)

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=args.batch_size,
                                      collate_fn=collate_fn)
        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=args.batch_size,
                                      collate_fn=collate_fn)

        train_datasets.append(train_dataset)
        train_dataloaders.append(train_dataloader)
        valid_datasets.append(valid_dataset)
        valid_dataloaders.append(valid_dataloader)

        models.append(model.to(device))

    optimizers = []
    schedulers = []
    loss_fns = []

    for i, train_dataloader in enumerate(train_dataloaders):
        t_total = len(train_dataloader) * args.epoch
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in models[i].named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0
            },
            {
                "params": [
                    p for n, p in models[i].named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=t_total)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        loss_fns.append(CrossEntropyLoss())

    set_seed(args.random_seed)

    best_vl_micro_f1s = [-1 for _ in range(len(tasks))]
    best_vl_macro_f1s = [-1 for _ in range(len(tasks))]
    epoch_evaluation_results = [[] for _ in range(len(tasks))]

    print("Training model for {}_{}_{}_{}-lr-{}_bs-{}_ml-{}_seed-{}_ws-{}_preprocess-{}_aug-{}_mp-{}".format(method, task, serialization, args.model_name, args.lr, args.batch_size, args.max_length, args.random_seed, args.window_size, args.preprocess, args.aug, args.mp))
    
    for epoch in range(args.epoch):
        for task_number, (task, model, train_dataset, valid_dataset, train_dataloader,
                valid_dataloader, optimizer, scheduler, loss_fn,
                epoch_evaluation_result) in enumerate(
                    zip(tasks, models, train_datasets, valid_datasets,
                        train_dataloaders, valid_dataloaders, optimizers,
                        schedulers, loss_fns, epoch_evaluation_results)):
            t1 = time()

            model.train()

            training_loss = 0.
            training_predictions = []
            training_labels = []

            validation_loss = 0.
            validation_predictions = []
            validation_labels = []

            for batch_idx, batch in enumerate(train_dataloader):
                # TODO: add serialization here
                batch_input_ids = batch["data"].T.to(device)
                batch_mask = batch["attention"].T.to(device)
                
                if (serialization == 'col_tabert') or (serialization == 'sum_neighbor') or (serialization == 'nomain_neighbor') or (serialization == 'tabert'):
                    batch_labels = batch["label"].float().to(device)
                    logits, = model(batch_input_ids, token_type_ids=None, attention_mask=batch_mask, return_dict=False)        
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, batch_labels.float())
                    
                    for pred in logits.cpu().detach():
                        sigmoid = torch.nn.Sigmoid()
                        probs = sigmoid(pred)
                        l_preds = np.zeros(probs.shape)
                        l_preds[np.where(probs >= 0.5)] = 1
                        training_predictions.append(l_preds)

                    #True labels:
                    training_labels += batch["label"].cpu().detach().numpy().tolist()
                else:
                    pass

                # Perform a backward pass to calculate the gradients.
                loss.backward()
                # Accumulate the training loss over all of the batches
                training_loss += loss.item()

                #Update parameters and take a step using the calculated gradient
                optimizer.step()
                #Update learning rate
                scheduler.step()
                #Clear previously calculated gradients
                model.zero_grad()

            training_loss /= (len(train_dataset) / args.batch_size)

            tr_micro_f1, tr_macro_f1, tr_class_f1, _ = f1_score_multilabel(training_labels, training_predictions)

            # Validation
            model.eval()
            for batch_idx, batch in enumerate(valid_dataloader):
                # TODO add serialization
                batch_input_ids = batch["data"].T.to(device)
                batch_mask = batch["attention"].T.to(device)
                
                if (serialization == 'col_tabert') or (serialization == 'sum_neighbor') or (serialization == 'nomain_neighbor') or (serialization == 'tabert'):
                    batch_labels = batch["label"].float().to(device)
                    logits, = model(batch_input_ids, token_type_ids=None, attention_mask=batch_mask, return_dict=False)        
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, batch_labels.float())
                    
                    for pred in logits.cpu().detach():
                        sigmoid = torch.nn.Sigmoid()
                        probs = sigmoid(pred)
                        l_preds = np.zeros(probs.shape)
                        l_preds[np.where(probs >= 0.5)] = 1
                        validation_predictions.append(l_preds)

                    validation_labels += batch["label"].cpu().detach().numpy().tolist()
                else:
                    pass

                validation_loss += loss.item()

            validation_loss /= (len(valid_dataset) / args.batch_size)

            vl_micro_f1, vl_macro_f1, vl_class_f1, _ = f1_score_multilabel(validation_labels, validation_predictions)

            #Mark highest micro-F1 and save model if it outputs highest F1
            if vl_micro_f1 > best_vl_micro_f1s[task_number]:
                best_vl_micro_f1s[task_number] = vl_micro_f1
                model_savepath = "model/{}_{}_{}_{}-lr-{}_bs-{}_ml-{}_seed-{}_ws-{}_preprocess-{}_aug-{}_mp-{}.pt".format(method, task, serialization, args.model_name, args.lr, args.batch_size, args.max_length, args.random_seed, args.window_size, args.preprocess, args.aug, args.mp)
                torch.save(model.state_dict(), model_savepath)

            epoch_evaluation_result.append([training_loss, tr_macro_f1, tr_micro_f1, validation_loss, vl_macro_f1, vl_micro_f1])

            t2 = time()

            print(
            "Epoch {} ({}): tr_loss={:.7f} tr_macro_f1={:.4f} tr_micro_f1={:.4f} "
            .format(epoch, task, training_loss, tr_macro_f1, tr_micro_f1),
            "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f} (Total time: {:.2f} sec.)"
            .format(validation_loss, vl_macro_f1, vl_micro_f1, (t2 - t1)), flush=True)
            
    if not os.path.exists('reports/'):
        print("{} not exists. Created".format('reports/'))
        os.makedirs("reports/")
        
    for task, epoch_evaluation in zip(tasks, epoch_evaluation_results):
        loss_info_df = pd.DataFrame(epoch_evaluation,
                                    columns=[
                                        "tr_loss", "tr_f1_macro_f1",
                                        "tr_f1_micro_f1", "vl_loss",
                                        "vl_f1_macro_f1", "vl_f1_micro_f1"
                                    ])

        loss_info_df.to_csv("reports/{}_{}_{}_{}-lr-{}_bs-{}_ml-{}_seed-{}_ws-{}_preprocess-{}_aug-{}_mp-{}_info.csv".format(method, task, serialization, args.model_name, args.lr, args.batch_size, args.max_length, args.random_seed, args.window_size, args.preprocess, args.aug, args.mp))

# prediction
    for task in tasks:

        model_path = "model/{}_{}_{}_{}-lr-{}_bs-{}_ml-{}_seed-{}_ws-{}_preprocess-{}_aug-{}_mp-{}.pt".format(method, task, serialization, args.model_name, args.lr, args.batch_size, args.max_length, args.random_seed, args.window_size, args.preprocess, args.aug, args.mp)
        print(model_path)

        # todo
        if (serialization == 'col_tabert') or (serialization == 'sum_neighbor') or (serialization == 'nomain_neighbor') or (serialization == 'tabert'):
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
            pass

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
            batch_input_ids = batch["data"].T.to(device)
            batch_mask = batch["attention"].T.to(device)
                
            if (serialization == 'col_tabert') or (serialization == 'sum_neighbor') or (serialization == 'nomain_neighbor') or (serialization == 'tabert'):
                batch_labels = batch["label"].float().to(device)
                logits, = model(batch_input_ids, token_type_ids=None, attention_mask=batch_mask, return_dict=False)        
                #loss_fct = BCEWithLogitsLoss()
                #loss = loss_fct(logits, batch_labels.float())

                for pred in logits.cpu().detach():
                    sigmoid = torch.nn.Sigmoid()
                    probs = sigmoid(pred)
                    l_preds = np.zeros(probs.shape)
                    l_preds[np.where(probs >= 0.5)] = 1
                    test_predictions.append(l_preds)

                test_labels += batch["label"].cpu().detach().numpy().tolist()
            else:
                pass

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