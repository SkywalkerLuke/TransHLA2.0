import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from models import Lora_ESM, TransHLA2_0_BIND, TransHLA2_0_IM, NoCNNModel, NoTransformerModel, NoCrossAttentionModel
from utils import set_seed, load_and_tokenize, test_loader_eval
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/TransHLA_train_version_8_clean.txt")
    parser.add_argument("--val_path", type=str, default="data/TransHLA_val_version_8_clean.txt")
    parser.add_argument("--test_path", type=str, default="data/TransHLA_test_version_8_clean.txt")
    parser.add_argument("--model_name", type=str, default="NoCNN", choices=["TransHLA2_0_BIND", "TransHLA2_0_IM", "NoCNN", "NoTransformer", "NoCrossAttention"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)

    (x_train_ep, x_train_hla, y_train,
     x_val_ep, x_val_hla, y_val,
     x_test_ep, x_test_hla, y_test, meta) = load_and_tokenize(
        args.train_path, args.val_path, args.test_path
    )
    test_df = meta["test_df"].reset_index(drop=True)

    model = build_model(args.model_name, args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)

    acc, roc_auc, mcc, f1, recall, precision, result, labels = test_loader_eval(
        x_test_ep, x_test_hla, y_test, args.batch_size, args.device, model
    )
    print(f"Test | acc {acc:.2f} | AUC {roc_auc:.4f} | MCC {mcc:.4f} | F1 {f1:.4f} | R {recall:.4f} | P {precision:.4f}")

    # ROC
    fpr, tpr, _ = roc_curve(np.array(labels), result[:, 1])
    roc_auc_plot = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_plot:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close()

    # PR
    precision_vals, recall_vals, _ = precision_recall_curve(labels, result[:, 1])
    auprc = average_precision_score(labels, result[:, 1])
    plt.figure()
    plt.step(recall_vals, precision_vals, where='post', label=f'AUPRC={auprc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    pr_path = os.path.join(args.output_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Confusion matrix
    predict_label = np.argmax(result, axis=1)
    cm = confusion_matrix(labels, predict_label)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure()
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j + 0.5, i + 0.6, f'\n{cm_norm[i, j]*100:.1f}%', ha='center', va='center', color=txt_color)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Export CSV
    out_csv = os.path.join(args.output_dir, f"results_{os.path.basename(args.test_path).replace('.txt','')}.csv")
    prediction_df = pd.DataFrame({
        'Pred_Prob_0': result[:, 0],
        'Pred_Prob_1': result[:, 1],
        'Pred_Label': predict_label,
        'True_Label': labels
    })
    export_df = pd.concat([test_df, prediction_df], axis=1)
    export_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}\nROC: {roc_path}\nPR: {pr_path}\nCM: {cm_path}")

if __name__ == "__main__":
    main()