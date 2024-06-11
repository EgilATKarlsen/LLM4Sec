import time
import pickle
import pandas as pd
from sklearn.cluster import KMeans as skKMeans
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, confusion_matrix, homogeneity_score,
                             completeness_score, v_measure_score)

class KMeans:
    def __init__(self, n_clusters=8, **kwargs):
        self.n_clusters = n_clusters
        self.model = skKMeans(n_clusters=n_clusters, **kwargs)
        self.centroids = None
        self.labels_ = None
        self.mapping = None
        self.metrics = {}

    def fit(self, data, y_train=None):
        start_time = time.time()
        
        self.model.fit(data)
        
        end_time = time.time()
        self.metrics['fit_time'] = end_time - start_time  # Store the time difference
        
        self.centroids = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        
        if y_train is not None:
            self.label_clusters(y_train)

    def predict(self, data):
        clusters = self.model.predict(data)
        if self.mapping:
            return [self.mapping.get(cluster, -1) for cluster in clusters]
        return clusters

    def label_clusters(self, y_train):
        scores_df = pd.DataFrame({
            'Cluster': self.labels_,
            'Label': y_train
        })
        # Determine the most frequent label for each cluster
        self.mapping = scores_df.groupby('Cluster')['Label'].apply(lambda x: x.mode().iloc[0]).to_dict()

    def save_instance(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_instance(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def evaluate(self, y_true, y_pred, split):
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

        # Other metrics (unchanged from your original code)
        data['Macro F1-Score'] = f1_score(y_true, y_pred, average='macro')
        data['Micro F1-Score'] = f1_score(y_true, y_pred, average='micro')
        data['Accuracy'] = accuracy_score(y_true, y_pred)
        data['F-Measure'] = data['Weighted F1-Score']
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        data['False Positive Rate'] = fp / (fp + tn)
        data['Homogeneity'] = homogeneity_score(y_true, y_pred)
        data['Completeness'] = completeness_score(y_true, y_pred)
        data['V-Measure'] = v_measure_score(y_true, y_pred)

        self.metrics[f'{split}'] = data

        return self.metrics

