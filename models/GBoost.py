import time
import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, confusion_matrix, homogeneity_score,
                             completeness_score, v_measure_score)
from sklearn.metrics import roc_curve, roc_auc_score, auc

class GradientBoost:
    def __init__(self, n_estimators=100, **kwargs):
        self.n_estimators = n_estimators
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42, verbose=1, **kwargs)
        self.metrics = {}

    def fit(self, data, y_train):
        start_time = time.time()

        self.model.fit(data, y_train)

        end_time = time.time()
        self.metrics['fit_time'] = end_time - start_time  # Store the time difference

    def predict(self, data):
        y_pred = self.model.predict(data)
        y_proba = self.model.predict_proba(data)
        return y_pred, y_proba

    def save_instance(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_instance(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def evaluate(self, y_true, y_pred, y_proba, split):
        data = {}

        unique_classes = sorted(list(set(y_true)))  # Extract unique class labels

        # Weighted Metrics
        data['Weighted F1-Score'] = f1_score(y_true, y_pred, average='weighted')
        data['Weighted Precision'] = precision_score(y_true, y_pred, average='weighted')
        data['Weighted Recall'] = recall_score(y_true, y_pred, average='weighted')

        # Class-based Metrics
        class_precisions = precision_score(y_true, y_pred, average=None).tolist()
        class_recalls = recall_score(y_true, y_pred, average=None).tolist()
        class_f1s = f1_score(y_true, y_pred, average=None).tolist()

        data['Class Precision'] = {cls: prec for cls, prec in zip(unique_classes, class_precisions)}
        data['Class Recall'] = {cls: recall for cls, recall in zip(unique_classes, class_recalls)}
        data['Class F1-Score'] = {cls: f1 for cls, f1 in zip(unique_classes, class_f1s)}

        # Other metrics
        data['Macro F1-Score'] = f1_score(y_true, y_pred, average='macro')
        data['Micro F1-Score'] = f1_score(y_true, y_pred, average='micro')
        data['Accuracy'] = accuracy_score(y_true, y_pred)
        data['F-Measure'] = data['Weighted F1-Score']

        # Confusion Matrix
        data['Confusion Matrix'] = confusion_matrix(y_true, y_pred).tolist()

        # Get the index for the "Anomalous" class
        positive_class_index = list(self.model.classes_).index('Normal')
        positive_class_probs = y_proba[:, positive_class_index]

        # ROC and AUC
        fpr, tpr, _ = roc_curve(y_true, positive_class_probs, pos_label="Normal")
        data['FPR'] = fpr.tolist()
        data['TPR'] = tpr.tolist()
        data['AUC'] = roc_auc_score(y_true, positive_class_probs)

        self.metrics[f'{split}'] = data

        return self.metrics