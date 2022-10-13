#Code adapted from DODUO: https://github.com/megagonlabs/doduo
import argparse
import json
import math
import random
from time import time
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import (
    collate_fn,
    #CTA
    CTASingleColumnDataset,
    CTAAllTableDataset,
    #CPA
    CPASingleColumnDataset,
    CPAAllTableDataset,
)

from model import BertForMultiOutputClassification, BertMultiPairPooler
from util import f1_score_multilabel, set_seed

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
    parser.add_argument("--method",
                        type=str,
                        nargs="+",
                        default=["cta"],
                        choices=[
                            "cta", "cpa", "doduo"
                        ],
                        help="Task names}")
    parser.add_argument("--serialization",
                    type=str,
                    nargs="+",
                    default="single-column",
                    choices=[
                        "single-column", "all-table"
                    ],
                    help="Serialization methods}")

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
            "cpa": 176
        }
    filepaths_task_dict = {
        "cta": "data/CTA/cta_lm.pkl",
        "cpa": "data/CPA/cpa_lm.pkl",
    }

    serialization_method_dict = {
        "cta": {
            "single-column": CTASingleColumnDataset,
            "all-table": CTAAllTableDataset
        },
        "cpa": {
            "single-column": CPASingleColumnDataset,
            "all-table": CPAAllTableDataset
        }
    }

    if not os.path.exists('model/'):
        print("{} not exists. Created".format(model/))
        os.makedirs(model/)


    #Tokenizer based on language model
    if 'roberta' in args.model_name:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name, use_fast=True)
        base_model = 'roberta'
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model_name, use_fast=True)
        base_model = 'bert'

    #If Doduo there are two models: one for CTA and one for CPA
    models = []
    train_datasets = []
    train_dataloaders = []
    valid_datasets = []
    valid_dataloaders = []

    for task in tasks:
        if serialization == 'single-column':
            #Choose model
            if 'roberta' in args.model_name:
                model_config = RobertaConfig.from_pretrained(args.model_name, num_labels=task_num_class_dict[task])
                model = RobertaForSequenceClassification(model_config)
            else:
                model_config = BertConfig.from_pretrained(args.model_name, num_labels=task_num_class_dict[task])
                model = BertForSequenceClassification(model_config)

            #Choose serialization
            dataset_serialization = serialization_method_dict[task][serialization]

        #Add more conditions when adding new serialization methods
        else:
            model = BertForMultiOutputClassification.from_pretrained(
                    args.model_name,
                    num_labels=task_num_class_dict[task],
                    output_attentions=False,
                    output_hidden_states=False,
                )

            dataset_serialization = serialization_method_dict[task][serialization]

            #Doduo:
            if task == "cpa":
                # Use column pair embeddings
                config = BertConfig.from_pretrained(args.model_name)
                model.bert.pooler = BertMultiPairPooler(config).to(device)


        #Load training and validation datasets and datasetloaders
        train_dataset = dataset_serialization(filepath=filepaths_task_dict[task],
                                        split="train",
                                        tokenizer=tokenizer,
                                        max_length=args.max_length,
                                        bert=base_model,
                                        device=device)

        valid_dataset = dataset_serialization(filepath=filepaths_task_dict[task],
                                    split="dev",
                                    tokenizer=tokenizer,
                                    max_length=args.max_length,
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=t_total)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        loss_fns.append(CrossEntropyLoss())

    set_seed(42)

    best_vl_micro_f1s = [-1 for _ in range(len(tasks))]
    best_vl_macro_f1s = [-1 for _ in range(len(tasks))]
    epoch_evaluation_results = [[] for _ in range(len(tasks))]

    print("Training model")
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

                if serialization == 'single-column':

                    #Retrive input ids, attention masks and labels for batch
                    batch_input_ids = batch["data"].T.to(device)
                    batch_mask = batch["attention"].T.to(device)
                    #For cross-entropy loss labels should not be vectors
                    batch_labels = torch.tensor([label.tolist().index(1) for label in batch["label"]]).to(device)
                    loss, logits = model(batch_input_ids, token_type_ids=None, attention_mask=batch_mask, labels=batch_labels, return_dict=False)

                    #Retrive predicted labels for batch
                    for pred in logits.argmax(axis=-1):
                        y = [0] * logits.shape[1]
                        y[pred] = 1
                        training_predictions.append(y)

                    #True labels:
                    training_labels += batch["label"].cpu().detach().numpy().tolist()

                else:
                    # Table serialization case
                    batch_labels = torch.tensor([label.tolist().index(1) for label in batch["label"] if 1 in label.tolist()]).to(device)
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
                            training_predictions.append(y)

                        training_labels += batch["label"].cpu().detach().numpy().tolist()
                        loss = loss_fn(filtered_logits, batch_labels)

                    else:
                        #Change
                        all_preds = []
                        for pred in filtered_logits.argmax(axis=-1):
                            y = [0] * filtered_logits.shape[1]
                            y[pred] = 1
                            all_preds.append(y)

                        all_labels = batch["label"].cpu().detach().numpy()
                        # Ignore the very first CLS token
                        idxes = np.where(all_labels > 0)[0]
                        #set_trace()

                        all_preds_filtered = [ pred for i, pred in enumerate(all_preds) if i in idxes ]
                        all_labels_filtered = [label.tolist() for label in batch["label"] if 1 in label.tolist()]

                        training_predictions += all_preds_filtered
                        #set_trace()
                        training_labels += all_labels_filtered
                        loss = loss_fn(filtered_logits, batch["label"].float())

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
                if serialization == 'single-column':

                    batch_input_ids = batch["data"].T.to(device)
                    batch_mask = batch["attention"].T.to(device)
                    #For cross-entropy loss labels should not be vectors
                    batch_labels = torch.tensor([label.tolist().index(1) for label in batch["label"]]).to(device)
                    loss, logits = model(batch_input_ids, token_type_ids=None, attention_mask=batch_mask, labels=batch_labels, return_dict=False)

                    for p in logits.argmax(axis=-1):
                        y = [0] * logits.shape[1]
                        y[p] = 1
                        validation_predictions.append(y)

                    validation_labels += batch["label"].cpu().detach().numpy().tolist()

                else:
                    batch_labels = torch.tensor([label.tolist().index(1) for label in batch["label"] if 1 in label.tolist()]).to(device)
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
                            validation_predictions.append(y)

                        validation_labels += batch["label"].cpu().detach().numpy().tolist()
                        loss = loss_fn(filtered_logits, batch_labels)

                    else:
                        all_preds = []
                        for pred in filtered_logits.argmax(axis=-1):
                            y = [0] * filtered_logits.shape[1]
                            y[pred] = 1
                            all_preds.append(y)

                        all_labels = batch["label"].cpu().detach().numpy()
                        # Ignore the very first CLS token
                        idxes = np.where(all_labels > 0)[0]

                        all_preds_filtered = [ pred for i, pred in enumerate(all_preds) if i in idxes ]
                        all_labels_filtered = [label.tolist() for label in batch["label"] if 1 in label.tolist()]
                        validation_predictions += all_preds_filtered

                        validation_labels += all_labels_filtered
                        loss = loss_fn(filtered_logits, batch["label"].float())

                validation_loss += loss.item()

            validation_loss /= (len(valid_dataset) / args.batch_size)

            vl_micro_f1, vl_macro_f1, vl_class_f1, _ = f1_score_multilabel(validation_labels, validation_predictions)

            #Mark highest micro-F1 and save model if it outputs highest F1
            if vl_micro_f1 > best_vl_micro_f1s[task_number]:

                best_vl_micro_f1s[task_number] = vl_micro_f1
                model_savepath = "model/{}_{}_{}_{}-bs{}-ml-{}.pt".format(method, task, serialization, args.model_name, args.batch_size, args.max_length)
                torch.save(model.state_dict(), model_savepath)

            epoch_evaluation_result.append([training_loss, tr_macro_f1, tr_micro_f1, validation_loss, vl_macro_f1, vl_micro_f1])

            t2 = time()

            print(
            "Epoch {} ({}): tr_loss={:.7f} tr_macro_f1={:.4f} tr_micro_f1={:.4f} "
            .format(epoch, task, training_loss, tr_macro_f1, tr_micro_f1),
            "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f} (Total time: {:.2f} sec.)"
            .format(validation_loss, vl_macro_f1, vl_micro_f1, (t2 - t1)))

    for task, epoch_evaluation in zip(tasks, epoch_evaluation_results):
        loss_info_df = pd.DataFrame(epoch_evaluation,
                                    columns=[
                                        "tr_loss", "tr_f1_macro_f1",
                                        "tr_f1_micro_f1", "vl_loss",
                                        "vl_f1_macro_f1", "vl_f1_micro_f1"
                                    ])

        loss_info_df.to_csv("model/{}_{}_{}_{}-bs{}-ml-{}_info.csv".format(method, task, serialization, args.model_name, args.batch_size, args.max_length))
