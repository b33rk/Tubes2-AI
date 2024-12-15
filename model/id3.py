from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from collections import Counter

class ID3(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0):
        """
        inisialisasi parameter:
        - max_depth: Kedalaman maksimum pohon (default: None, tidak dibatasi).
        - min_samples_split: Jumlah minimum sampel untuk membagi node (default: 2).
        - min_impurity_decrease: Penurunan impurity minimum agar node dipecah (default: 0).
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None # untuk menyimpan pohon yang terbentuk
        self.feature_names = None # menyimpan nama fitur

    def get_params(self, deep=True):
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_impurity_decrease": self.min_impurity_decrease
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @staticmethod
    def entropy(y):
        """
        menghitung entropy target variable y.
        - parameter:
          - y: array target kelas.
        - output: nilai entropy (tingkat impurity).
        """
        counts = Counter(y) # hitung kemunculan setiap kelas
        probabilities = [count / len(y) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probabilities if p > 0)

    def information_gain(self, X_column, y):
        """
        menghitung information gain untuk fitur tertentu.
        - parameter:
          - X_column: kolom fitur yang sedang diuji.
          - y: array target kelas.
        - output: nilai information gain.
        """
        total_impurity = self.entropy(y) # impurity sebelum split

        values, counts = np.unique(X_column, return_counts=True)
        # menghitung impurity setelah split
        weighted_impurity = sum(
            (counts[i] / len(X_column)) * 
            (self.entropy(y[X_column == values[i]]))
            for i in range(len(values))
        )
        return total_impurity - weighted_impurity

    def best_split(self, X, y):
        """
        menentukan fitur terbaik untuk melakukan split.
        - parameter:
          - X: data fitur (2D array).
          - y: target kelas.
        - output: indeks fitur terbaik dan threshold terbaik.
        """
        max_info_gain = -np.inf # nilai maksimum information gain
        best_feature_idx = None
        best_threshold = None

        for i in range(X.shape[1]):
            X_column = X[:, i]
            
            # cek tipe data fitur
            if np.issubdtype(X_column.dtype, np.number): # fitur numerik
                thresholds = [
                    np.percentile(X_column, q) 
                    for q in [25, 50, 75]
                ]
                for threshold in thresholds:
                    left = X_column <= threshold
                    right = X_column > threshold
                    
                    # hanya akan split jika jumlah sampel di kedua sisi >= min_samples_split
                    if (sum(left) >= self.min_samples_split and 
                        sum(right) >= self.min_samples_split):
                        gain = self.information_gain(left, y)
                        
                        if gain > max_info_gain and gain > self.min_impurity_decrease:
                            max_info_gain = gain
                            best_feature_idx = i
                            best_threshold = threshold
            else: # fitur kategorikal
                gain = self.information_gain(X_column, y)
                if gain > max_info_gain and gain > self.min_impurity_decrease:
                    max_info_gain = gain
                    best_feature_idx = i
                    best_threshold = None

        return best_feature_idx, best_threshold

    def build_tree(self, X, y, depth=0):
        """
        membangun pohon keputusan secara rekursif.
        - parameter:
          - X: data fitur.
          - y: target kelas.
          - depth: Kedalaman saat ini.
        - output: Pohon dalam bentuk dictionary.
        """
        # cek kondisi berhenti
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples_split or \
           len(np.unique(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        # find best split
        best_feature_idx, best_threshold = self.best_split(X, y)

        # if no good split found, return majority class
        if best_feature_idx is None:
            return Counter(y).most_common(1)[0][0]

        # create tree node
        tree = {
            'feature': self.feature_names[best_feature_idx],
            'threshold': best_threshold,
            'children': {}
        }

        # create splits
        if best_threshold is not None:
            # continuous feature split
            left_mask = X[:, best_feature_idx] <= best_threshold
            right_mask = ~left_mask

            left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
            right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

            tree['children']['left'] = left_subtree
            tree['children']['right'] = right_subtree
        else:
            # categorical feature split
            unique_values = np.unique(X[:, best_feature_idx])
            for value in unique_values:
                mask = X[:, best_feature_idx] == value
                subtree = self.build_tree(X[mask], y[mask], depth + 1)
                tree['children'][value] = subtree

        return tree

    def fit(self, X, y, feature_names=None):
        """melatih model"""
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.feature_names = feature_names
        y = np.array(y).flatten()
        self.majority_class = Counter(y).most_common(1)[0][0]
        self.tree = self.build_tree(X, y)
        return self

    def predict_instance(self, instance):
        """memprediksi kelas untuk satu instance."""
        node = self.tree

        while isinstance(node, dict):
            feature = node['feature']
            threshold = node['threshold']
            
            feature_idx = self.feature_names.index(feature)
            feature_value = instance[feature_idx]

            if threshold is not None:
                # continuous feature
                if feature_value <= threshold:
                    node = node['children']['left']
                else:
                    node = node['children']['right']
            else:
                # categorical feature
                if feature_value in node['children']:
                    node = node['children'][feature_value]
                else:
                    return self.majority_class

        return node

    def predict(self, X):
        """memprediksi kelas untuk banyak instance."""
        return np.array([self.predict_instance(instance) for instance in X])