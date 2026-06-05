"""Machine learning classification for drug response prediction"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class DrugResponsePredictor:
    """Predict platinum/RelB-inhibitor drug response"""
    
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100)
    
    def train(self, expression_df, response_labels):
        """Train classifier"""
        X = self.scaler.fit_transform(expression_df)
        self.model.fit(X, response_labels)
        return self
    
    def predict(self, expression_df):
        """Predict response (0=resistant, 1=sensitive)"""
        X = self.scaler.transform(expression_df)
        return self.model.predict(X)
    
    def predict_proba(self, expression_df):
        """Predict probability of response"""
        X = self.scaler.transform(expression_df)
        return self.model.predict_proba(X)[:, 1]
    
    def cross_validate(self, expression_df, response_labels, cv=5):
        """Cross-validation scoring"""
        X = self.scaler.fit_transform(expression_df)
        scores = cross_val_score(self.model, X, response_labels, 
                                cv=StratifiedKFold(n_splits=cv),
                                scoring='roc_auc')
        return scores
    
    def feature_importance(self):
        """Get feature importance (for RandomForest)"""
        if self.model_type == 'rf':
            return pd.Series(self.model.feature_importances_).sort_values(ascending=False)
        return None

print("✓ DrugResponsePredictor loaded")
