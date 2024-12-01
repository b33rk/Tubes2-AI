import numpy as np
import pandas as pd
from collections import Counter

class ID3:
    def __init__(self, max_depth=None):
        self.tree = None
        self.max_depth = max_depth

    @staticmethod
    def entropy(y):
        """Calculate entropy of the target variable."""
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probabilities if p > 0)

    def information_gain(self, X_column, y):
        """Calculate information gain for a feature."""
        total_entropy = self.entropy(y)
        values, counts = np.unique(X_column, return_counts=True)
        weighted_entropy = sum(
            (counts[i] / len(X_column)) * self.entropy(y[X_column == values[i]])
            for i in range(len(values))
        )
        return total_entropy - weighted_entropy

    def best_split(self, X, y, feature_names):
        """Determine the best feature to split on, considering thresholds for continuous features."""
        max_info_gain = -1
        best_feature = None
        best_threshold = None

        for i, feature_name in enumerate(feature_names):
            X_column = X[:, i]
            if np.issubdtype(X_column.dtype, np.number):
                # Continuous feature
                thresholds = np.unique(X_column)
                for threshold in thresholds[:-1]:
                    left = X_column <= threshold
                    right = X_column > threshold
                    gain = self.information_gain(left, y)
                    if gain > max_info_gain:
                        max_info_gain = gain
                        best_feature = feature_name
                        best_threshold = threshold
            else:
                # Categorical feature
                gain = self.information_gain(X_column, y)
                if gain > max_info_gain:
                    max_info_gain = gain
                    best_feature = feature_name
                    best_threshold = None

        return best_feature, best_threshold

    def build_tree(self, X, y, feature_names, depth=0):
        """Recursively build the decision tree without pruning."""
        if len(np.unique(y)) == 1:
            return y[0]  # Pure node

        if len(feature_names) == 0:
            return Counter(y).most_common(1)[0][0]  # Majority vote

        best_feature, best_threshold = self.best_split(X, y, feature_names)

        if best_feature is None:
            return Counter(y).most_common(1)[0][0]  # Majority vote

        feature_idx = feature_names.index(best_feature)
        tree = {best_feature: {}}

        if best_threshold is not None:
            # Continuous feature
            left_indices = X[:, feature_idx] <= best_threshold
            right_indices = X[:, feature_idx] > best_threshold

            tree[best_feature]["<= {:.2f}".format(best_threshold)] = self.build_tree(
                X[left_indices], y[left_indices], feature_names, depth + 1
            )
            tree[best_feature]["> {:.2f}".format(best_threshold)] = self.build_tree(
                X[right_indices], y[right_indices], feature_names, depth + 1
            )
        else:
            # Categorical feature
            unique_values = np.unique(X[:, feature_idx])
            for value in unique_values:
                subset_indices = X[:, feature_idx] == value
                subtree = self.build_tree(
                    X[subset_indices], y[subset_indices], feature_names, depth + 1
                )
                tree[best_feature][value] = subtree

        return tree


    def fit(self, X, y, feature_names):
        """Fit the decision tree to the data."""
        self.tree = self.build_tree(X, y, feature_names)

    def predict_instance(self, instance, tree):
        """Predict the target value for a single instance."""
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        subtree = None

        if feature in instance:
            value = instance[feature]
            subtree = tree[feature].get(value, None)
            if subtree is None:  # Handle continuous feature splits
                for key in tree[feature]:
                    if key.startswith("<=") and float(value) <= float(key[3:]):
                        subtree = tree[feature][key]
                    elif key.startswith(">") and float(value) > float(key[2:]):
                        subtree = tree[feature][key]
                    if subtree:
                        break

        if subtree is None:
            return None

        return self.predict_instance(instance, subtree)

    def predict(self, X, feature_names):
        """Predict the target values for multiple instances."""
        predictions = []
        for instance in X:
            instance_dict = {feature_names[i]: instance[i] for i in range(len(feature_names))}
            predictions.append(self.predict_instance(instance_dict, self.tree))
        return predictions

    def print_tree(self, tree=None, feature_names=None, encoders=None, indent=""):
        """Print the full tree in a readable format with decoded feature values."""
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            # Decode target variable
            decoded_value = encoders["Aktivitas"].inverse_transform([tree])[0] if encoders else tree
            print(indent + f"-> {decoded_value}")
            return

        for key, subtree in tree.items():
            if isinstance(subtree, dict):
                for subkey, subsubtree in subtree.items():
                    # Handle continuous thresholds and categorical decoding
                    if encoders and key in encoders and not any(op in subkey for op in ["<=", ">"]):
                        decoded_subkey = encoders[key].inverse_transform([int(subkey)])[0]
                        print(indent + f"If {key} = {decoded_subkey}:")
                    else:
                        # Print for thresholds (continuous features)
                        print(indent + f"If {key} {subkey}:")
                    self.print_tree(subsubtree, feature_names, encoders, indent + "    ")
            else:
                print(indent + f"If {key}:")
                self.print_tree(subtree, feature_names, encoders, indent + "    ")


# CONTOH NGETES ONLY
data = {
    "Deadline": ["Urgent", "Urgent", "Dekat", "Tidak ada", "Tidak ada", "Tidak ada", "Dekat", "Dekat", "Dekat", "Urgent"],
    "Ada Hangout?": ["ya", "Tidak", "ya", "Ya", "Tidak", "Ya", "Tidak", "Tidak", "Ya", "Tidak"],
    "Malas?": ["Ya", "Ya", "Ya", "Tidak", "Ya", "Tidak", "Tidak", "Ya", "Ya", "Tidak"],
    "Aktivitas": ["Kumpul-kumpul", "Belajar", "Kumpul-kumpul", "Kumpul-kumpul", "Jalan-jalan ke mall", "Kumpul-kumpul", "Belajar", "Nonton TV", "Kumpul-kumpul", "Belajar"]
}

df = pd.DataFrame(data)

feature_columns = ["Deadline", "Ada Hangout?", "Malas?"]
target_column = "Aktivitas"

# encode categorical data
from sklearn.preprocessing import LabelEncoder
encoders = {col: LabelEncoder().fit(df[col]) for col in feature_columns + [target_column]}
df_encoded = df.apply(lambda col: encoders[col.name].transform(col))

# split features and target
X = df_encoded[feature_columns].values
y = df_encoded[target_column].values
feature_names = feature_columns

# train the ID3 model
model = ID3(max_depth=3)
model.fit(X, y, feature_names)

# print the decision tree 4 testing purposes
print("Decision Tree:")
model.print_tree(feature_names=feature_names, encoders=encoders)
