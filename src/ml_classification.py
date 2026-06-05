"""Machine learning classification for drug response prediction"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DrugResponsePredictor:
    """Predict platinum/RelB-inhibitor drug response"""

    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.is_trained = False

        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"model_type must be 'logistic' or 'rf', got '{model_type}'")

    def train(self, expression_df, response_labels):
        """
        Train classifier on expression data.

        Parameters:
        -----------
        expression_df : pd.DataFrame or np.ndarray (samples × genes)
        response_labels : array-like (0=resistant, 1=sensitive)

        Returns:
        --------
        self
        """
        X = np.array(expression_df)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, response_labels)
        self.is_trained = True
        return self

    def predict(self, expression_df):
        """
        Predict binary drug response (0=resistant, 1=sensitive).

        Returns:
        --------
        np.ndarray of shape (n_samples,)
        """
        X = np.array(expression_df)
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, expression_df):
        """
        Predict probability of sensitive response.

        Returns:
        --------
        np.ndarray of shape (n_samples,)
        """
        X = np.array(expression_df)
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def cross_validate(self, expression_df, response_labels, cv=5):
        """
        Stratified k-fold cross-validation.

        Returns:
        --------
        pd.DataFrame with 'roc_auc' column of per-fold scores
        """
        X = np.array(expression_df)
        # Fit scaler on full data for consistent transform across folds
        X_scaled = self.scaler.fit_transform(X)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        raw_scores = cross_val_score(
            self.model, X_scaled, response_labels,
            cv=skf, scoring='roc_auc'
        )
        return pd.DataFrame({'roc_auc': raw_scores})

    def feature_importance(self):
        """
        Return feature importances (RF only) or absolute coefficients (logistic).

        Returns:
        --------
        pd.Series sorted descending
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained — call train() first")

        if self.model_type == 'rf':
            importances = self.model.feature_importances_
        else:
            importances = np.abs(self.model.coef_[0])

        return pd.Series(importances).sort_values(ascending=False)


print("✓ DrugResponsePredictor loaded")
