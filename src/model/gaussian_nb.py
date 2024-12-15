import numpy as np
import pandas as pd 

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.variances = {}
        self.priors = {}
    
    def fit(self, X, y):
        """
        Train the Gaussian Naive Bayes model
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        y = np.array(y)

        self.classes = np.unique(y)
        for cls in self.classes:
            # boolean mask for the current class
            mask = (y == cls)
            
            # mask is used to select rows for the current class
            X_cls = X.iloc[mask] if isinstance(X, pd.DataFrame) else X[mask]
            
            self.means[cls] = np.mean(X_cls, axis=0)
            self.variances[cls] = np.var(X_cls, axis=0)
            self.variances[cls] = np.where(self.variances[cls] == 0, 1e-9, self.variances[cls])  # Prevent zero variance
            self.priors[cls] = X_cls.shape[0] / X.shape[0]

    def _gaussian_pdf(self, x, mean, variance):
        """
        Calculate the Gaussian probability density function for a given value using the gaussian distribution formula
        """
        variance = max(variance, 1e-9)  # Add a small value to variance
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
        return (1 / np.sqrt(2 * np.pi * variance)) * exponent

    def _calculate_likelihood(self, X, cls):
        """
        Calculate the likelihood for a given class (probability data given class)
        """
        mean = self.means[cls]
        variance = self.variances[cls]
        likelihoods = np.apply_along_axis(
            lambda x: np.prod([self._gaussian_pdf(x[i], mean[i], variance[i]) for i in range(len(mean))]), 
            axis=1, 
            arr=X
        )
        return likelihoods

    def predict(self, X):
        """
        Predict the class labels for the input samples
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        posteriors = []
        for cls in self.classes:
            prior = self.priors[cls]
            likelihood = self._calculate_likelihood(X, cls)
            posterior = prior * likelihood
            posteriors.append(posterior)
        
        posteriors = np.array(posteriors).T 
        # print("Posteriors:\n", posteriors)
        predictions = np.argmax(posteriors, axis=1)
        return self.classes[predictions]
