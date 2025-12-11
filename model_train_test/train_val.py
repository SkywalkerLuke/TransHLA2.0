import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from models import Lora_ESM, TransHLA2_0_BIND, TransHLA2_0_IM, NoCNNModel, NoTransformerModel, NoCrossAttentionModel, reinit_classifier
from utils import (
    set_seed, load_and_tokenize, unbalanced_addbatch, addbatch,
    get_val_loss, test_loader_eval
)

def build_model(model_name: str, device: str):
    if model_name == "TransHLA2_0_IM":
        model = TransHLA2_0_IM()
    else:
        backbone = Lora_ESM()
        if model_name == "TransHLA2_0_BIND":
            model = TransHLA2_0_BIND(backbone)
        elif model_name == "NoCNN":
            model = NoCNNModel(backbone)
        elif model_name == "NoTransformer":
            model = NoTransformerModel(backbone)
        elif model_name == "NoCrossAttention":
            model = NoCrossAttentionModel(backbone)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
    return model.to(device)

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for step, (epitope_inputs, hla_inputs, labels) in enumerate(loader):
        epitope_inputs = epitope_inputs.to(device)
        hla_inputs = hla_inputs.to(device)
        labels = labels.to(device)

        outputs, _ = model(epitope_inputs, hla_inputs)
        loss = get_val_loss(outputs, labels, criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / (step + 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/TransHLA_train_version_8_clean.txt")
    parser.add_argument("--val_path", type=str, default="data/TransHLA_val_version_8_clean.txt")
    parser.add_argument("--test_path", type=str, default="data/TransHLA_test_version_8_clean.txt")
    parser.add_argument("--model_name", type=str, default="NoCNN", choices=["TransHLA2_0_BIND", "TransHLA2_0_IM", "NoCNN", "NoTransformer", "NoCrossAttention"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=2.5e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced", action="store_true", help="Use simple shuffle instead of weighted sampler")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_prefix", type=str, default="best_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    (x_train_ep, x_train_hla, y_train,
     x_val_ep, x_val_hla, y_val,
     x_test_ep, x_test_hla, y_test, meta) = load_and_tokenize(
        args.train_path, args.val_path, args.test_path
    )

    model = build_model(args.model_name, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if args.balanced:
        train_loader = addbatch(x_train_ep, x_train_hla, y_train, args.batch_size, shuffle=True)
    else:
        train_loader = unbalanced_addbatch(x_train_ep, x_train_hla, y_train, args.batch_size)

    best_auc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, args.device, criterion, optimizer)

        acc, auc, mcc, f1, recall, precision, _, _ = test_loader_eval(
            x_val_ep, x_val_hla, y_val, args.batch_size, args.device, model
        )

        print(f"Epoch {epoch:03d} | loss {avg_loss:.4f} | val_acc {acc:.2f} | AUC {auc:.4f} | MCC {mcc:.4f} | F1 {f1:.4f} | R {recall:.4f} | P {precision:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(args.save_dir, args.save_prefix))
            print("Best model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 最终在验证集上报告
    acc, auc, mcc, f1, recall, precision, _, _ = test_loader_eval(
        x_val_ep, x_val_hla, y_val, args.batch_size, args.device, model
    )
    print(f"Best on Val | acc {acc:.2f} | AUC {auc:.4f} | MCC {mcc:.4f} | F1 {f1:.4f} | R {recall:.4f} | P {precision:.4f}")

if __name__ == "__main__":
    main()