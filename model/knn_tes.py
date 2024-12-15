import numpy as np
from collections import Counter
import time

class KNN:
    def __init__(self, distance='euclidean', n_neighbors=5, batch_size=1000, verbose=True):
        self.distance = distance
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        if self.verbose:
            print(f"Model fitted with {len(self.X_train)} training samples.")

    def _compute_distance(self, X_batch):
        if self.distance == 'euclidean':
            return np.linalg.norm(self.X_train - X_batch[:, np.newaxis], axis=2)
        elif self.distance == 'manhattan':
            return np.sum(np.abs(self.X_train - X_batch[:, np.newaxis]), axis=2)
        elif self.distance == 'minkowski':
            p = 3
            return np.sum(np.abs(self.X_train - X_batch[:, np.newaxis])**p, axis=2)**(1/p)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

    def predict(self, X):
        X = np.array(X)
        predictions = []
        total_batches = (X.shape[0] + self.batch_size - 1) // self.batch_size

        for idx, start in enumerate(range(0, X.shape[0], self.batch_size), 1):
            if self.verbose:
                print(f"Processing batch {idx}/{total_batches}...")
            start_time = time.time()
            X_batch = X[start:start + self.batch_size]
            distances = self._compute_distance(X_batch)
            
            for d in distances:
                k_indices = np.argsort(d)[:self.n_neighbors]
                k_nearest_labels = [tuple(self.y_train[i]) if isinstance(self.y_train[i], np.ndarray) else self.y_train[i] for i in k_indices]
                most_common = Counter(k_nearest_labels).most_common(1)[0][0]
                predictions.append(most_common)

            if self.verbose:
                elapsed_time = time.time() - start_time
                print(f"Batch {idx} processed in {elapsed_time:.2f} seconds.")
        return np.array(predictions)
    
    