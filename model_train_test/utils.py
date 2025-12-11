import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score, recall_score, precision_score,
    roc_curve, auc, confusion_matrix, average_precision_score, precision_recall_curve
)
import pandas as pd
from transformers import AutoTokenizer

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

class TCRDataset(TensorDataset):
    def __init__(self, epitope_data, hla_data, labels):
        self.epitope_data = epitope_data
        self.hla_data = hla_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.epitope_data[idx], self.hla_data[idx], self.labels[idx]

def addbatch(epitope_data, hla_data, labels, batchsize, shuffle=True):
    dataset = TCRDataset(epitope_data, hla_data, labels)
    return DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)

def unbalanced_addbatch(epitope_data, hla_data, labels, batchsize):
    dataset = TCRDataset(epitope_data, hla_data, labels)
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    unique, counts = np.unique(labels_np, return_counts=True)
    weight_per_class = {cls: 1.0 / c for cls, c in zip(unique, counts)}
    samples_weight = np.array([weight_per_class[int(t)] for t in labels_np])
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).float(), len(samples_weight), replacement=True)
    return DataLoader(dataset, batch_size=batchsize, sampler=sampler)

def pad_inner_lists_to_length(outer_list, target_length=16, pad_id=1):
    for inner_list in outer_list:
        padding_length = target_length - len(inner_list)
        if padding_length > 0:
            inner_list.extend([pad_id] * padding_length)
        elif padding_length < 0:
            del inner_list[target_length:]  # truncate if too long (safety)
    return outer_list

def get_entropy(probs: torch.Tensor):
    p = probs.mean(0)
    return -(p * torch.log2(p + 1e-12)).sum(0, keepdim=True)

def get_cond_entropy(probs: torch.Tensor):
    return -(probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)

def get_val_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, 2), label.view(-1)).float().mean()
    loss = (loss - 0.04).abs() + 0.04
    logits = F.softmax(logits, dim=1)
    sum_loss = loss + get_entropy(logits) - get_cond_entropy(logits)
    return sum_loss[0]

def get_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, 2), label.view(-1)).float().mean()
    loss = (loss - 0.04).abs() + 0.04
    return loss

def test_loader_eval(test_epitope, test_hla, test_labels, batchsize, device, model):
    model.eval()
    correct = 0
    result_list, labels_list, predicted_list = [], [], []

    loader = addbatch(test_epitope, test_hla, test_labels, batchsize, shuffle=False)
    with torch.no_grad():
        for epitope_inputs, hla_inputs, labels in loader:
            epitope_inputs = epitope_inputs.to(device)
            hla_inputs = hla_inputs.to(device)
            labels = labels.to(device)

            result, _ = model(epitope_inputs, hla_inputs)
            result_list.append(result.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
            _, predicted = torch.max(result, 1)
            predicted_list.append(predicted.cpu().numpy())
            correct += (predicted == labels).sum().item()

    result_arr = np.concatenate(result_list)
    labels_arr = np.concatenate(labels_list)
    preds_arr = np.concatenate(predicted_list)
    acc = 100.0 * correct / labels_arr.shape[0]

    auc_score = roc_auc_score(labels_arr, result_arr[:, 1])
    mcc = matthews_corrcoef(preds_arr, labels_arr)
    f1 = f1_score(preds_arr, labels_arr)
    recall = recall_score(labels_arr, preds_arr)
    precision = precision_score(labels_arr, preds_arr)
    return acc, auc_score, mcc, f1, recall, precision, result_arr, labels_arr

def load_and_tokenize(train_path, val_path, test_path,
                      tokenizer_name="facebook/esm2_t33_650M_UR50D",
                      pep_len=16, hla_len=36):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _read(path):
        df = pd.read_csv(path, sep="\t")
        df['peptide'] = df['peptide'].str.upper()
        df['pseudosequence'] = df['pseudosequence'].replace(0, '<pad>' * 34)
        return df

    train = _read(train_path)
    val = _read(val_path)
    test = _read(test_path)

    def _encode(series, pad_len):
        token_ids = tokenizer(series.tolist())['input_ids']
        token_ids = pad_inner_lists_to_length(token_ids, target_length=pad_len, pad_id=1)
        return torch.tensor(token_ids, dtype=torch.long)

    x_train_ep = _encode(train['peptide'], pep_len)
    x_val_ep = _encode(val['peptide'], pep_len)
    x_test_ep = _encode(test['peptide'], pep_len)

    x_train_hla = _encode(train['pseudosequence'], hla_len)
    x_val_hla = _encode(val['pseudosequence'], hla_len)
    x_test_hla = _encode(test['pseudosequence'], hla_len)

    y_train = torch.tensor(train['label'].astype('int64').values)
    y_val = torch.tensor(val['label'].astype('int64').values)
    y_test = torch.tensor(test['label'].astype('int64').values)

    meta = {"train_df": train, "val_df": val, "test_df": test}
    return (x_train_ep, x_train_hla, y_train,
            x_val_ep, x_val_hla, y_val,
            x_test_ep, x_test_hla, y_test, meta)