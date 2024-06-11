import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, v_measure_score, homogeneity_score, completeness_score,
                             roc_auc_score, roc_curve)
from tqdm import tqdm
import time

def evaluate(model, tokenizer, test_dataset, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    labels = []
    preds = []
    probabilities = []

    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty")

    start_time = time.time()
    with torch.no_grad(), tqdm(total=len(test_dataset), desc="Evaluating") as pbar:
        for example in test_dataset:
            inputs = tokenizer(example['log'], truncation=True, padding=True, max_length=512, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            batch_labels = torch.tensor(example['label']).to(device)  # Convert label to tensor

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            if logits.dim() == 1:
                batch_preds = logits.detach().cpu().numpy()
                batch_probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            else:
                batch_preds = logits.argmax(dim=-1).detach().cpu().numpy()
                batch_probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()[:, 1]

            labels.append(batch_labels.cpu().numpy())
            preds.append(batch_preds)
            probabilities.append(batch_probabilities)

            pbar.update(1)

    end_time = time.time()
    execution_time = (end_time - start_time)

    labels = np.ravel(labels)
    preds = np.concatenate(preds)
    probabilities = np.concatenate(probabilities)

    if len(labels) == 0 or len(preds) == 0:
        raise ValueError("No valid examples found in the test dataset")

    accuracy = (labels == preds).mean()

    # Weighted metrics
    f1_weighted = f1_score(labels, preds, average='weighted')
    precision_weighted = precision_score(labels, preds, average='weighted')
    recall_weighted = recall_score(labels, preds, average='weighted')

    # Macro metrics
    f1_macro = f1_score(labels, preds, average='macro')
    precision_macro = precision_score(labels, preds, average='macro')
    recall_macro = recall_score(labels, preds, average='macro')

    # Micro metrics
    f1_micro = f1_score(labels, preds, average='micro')
    precision_micro = precision_score(labels, preds, average='micro')
    recall_micro = recall_score(labels, preds, average='micro')

    # Class-based metrics
    unique_classes = sorted(list(set(labels)))
    class_precisions = precision_score(labels, preds, average=None).tolist()
    class_recalls = recall_score(labels, preds, average=None).tolist()
    class_f1s = f1_score(labels, preds, average=None).tolist()

    # Other metrics
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    false_positive_rate = fp / (fp + tn)
    homogeneity = homogeneity_score(labels, preds)
    completeness = completeness_score(labels, preds)
    v_measure = v_measure_score(labels, preds)
    auc = roc_auc_score(labels, probabilities)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()

    eval_result = {
        "accuracy": accuracy,
        "f1_score_weighted": f1_weighted,
        "f1_score_macro": f1_macro,
        "f1_score_micro": f1_micro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "class_precision": {cls: prec for cls, prec in zip(unique_classes, class_precisions)},
        "class_recall": {cls: recall for cls, recall in zip(unique_classes, class_recalls)},
        "class_f1": {cls: f1 for cls, f1 in zip(unique_classes, class_f1s)},
        "false_positive_rate": false_positive_rate,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "time": execution_time,
        "auc": auc,
        "labels": labels.tolist(),
        "probabilities": probabilities.tolist(),
        "preds": preds.tolist()
    }

    return eval_result