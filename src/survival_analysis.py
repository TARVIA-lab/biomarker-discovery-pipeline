"""Survival analysis module for prognostic power assessment"""
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import spearmanr
from tqdm import tqdm

class SurvivalAnalyzer:
    """Cox regression and Kaplan-Meier analysis"""
    
    def __init__(self):
        self.cph = CoxPHFitter()
        self.km = KaplanMeierFitter()
    
    def univariate_cox(self, expression_df, time_col, event_col, metadata_df):
        """
        Run univariate Cox regression for each gene
        
        Returns:
        --------
        results : pd.DataFrame
            gene, HR, SE, p_value, log2_HR, significant
        """
        results = []
        
        for gene in tqdm(expression_df.columns, desc="Cox regression"):
            try:
                data = metadata_df.copy()
                data[gene] = expression_df[gene]
                
                self.cph.fit(data[[gene, time_col, event_col]], 
                           duration_col=time_col, 
                           event_col=event_col)
                
                hr = self.cph.hazard_ratios_[0]
                ci_lower = self.cph.confidence_interval_hazard_ratios_.iloc[0, 0]
                ci_upper = self.cph.confidence_interval_hazard_ratios_.iloc[0, 1]
                p_value = self.cph.summary['p'].values[0]
                
                results.append({
                    'gene': gene,
                    'hazard_ratio': hr,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'log2_HR': np.log2(hr),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
            except:
                continue
        
        return pd.DataFrame(results).sort_values('p_value')
    
    def kaplan_meier(self, expression_series, time_col, event_col, metadata_df, cutoff=None):
        """
        Generate Kaplan-Meier curves for high vs low expression
        """
        if cutoff is None:
            cutoff = expression_series.median()
        
        data = metadata_df.copy()
        data['expression'] = expression_series
        data['group'] = (data['expression'] > cutoff).astype(int)
        
        return data, cutoff

print("✓ SurvivalAnalyzer loaded")
