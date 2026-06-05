"""Survival analysis module for prognostic power assessment"""
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import norm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SurvivalAnalyzer:
    """Cox regression and Kaplan-Meier analysis"""

    def __init__(self):
        self.km = KaplanMeierFitter()

    def univariate_cox(self, expression_df, time_col, event_col, metadata_df):
        """
        Run univariate Cox regression for each gene

        Parameters:
        -----------
        expression_df : pd.DataFrame
            Samples × Genes expression matrix
        time_col : str
            Name of time column in metadata_df
        event_col : str
            Name of event column in metadata_df
        metadata_df : pd.DataFrame
            Metadata with time_col and event_col

        Returns:
        --------
        results : list of dicts
            gene, hazard_ratio, ci_lower, ci_upper, log2_HR, p_value, significant
        """
        results = []
        error_count = 0
        success_count = 0

        for gene in tqdm(expression_df.columns, desc="Cox regression"):
            try:
                # Prepare data
                data = pd.DataFrame({
                    time_col: metadata_df[time_col].values,
                    event_col: metadata_df[event_col].values,
                    gene: expression_df[gene].values
                })

                # Remove NaN
                data = data.dropna()
                if len(data) < 3 or data[event_col].sum() == 0:
                    continue

                # Fit Cox model with a fresh fitter for each gene
                cph = CoxPHFitter()
                cph.fit(data, duration_col=time_col, event_col=event_col, show_progress=False)

                # Extract statistics
                params = cph.params_
                se = cph.standard_errors_
                ci = cph.confidence_intervals_  # CI for coefficients

                if len(params) == 0 or len(se) == 0:
                    continue

                coef = float(params.iloc[0])
                hr = np.exp(coef)
                # Convert CI from log scale to HR scale
                ci_lower = float(np.exp(ci.iloc[0, 0]))
                ci_upper = float(np.exp(ci.iloc[0, 1]))

                # P-value
                se_val = float(se.iloc[0])
                if se_val > 0:
                    z = coef / se_val
                    p_val = 2 * (1 - norm.cdf(abs(z)))
                else:
                    p_val = 1.0

                results.append({
                    'gene': gene,
                    'hazard_ratio': float(hr),
                    'ci_lower': float(ci_lower),
                    'ci_upper': float(ci_upper),
                    'log2_HR': float(np.log2(hr)) if hr > 0 else 0.0,
                    'p_value': float(p_val),
                    'significant': p_val < 0.05
                })

            except Exception as e:
                error_count += 1
                pass

        # Return sorted by p-value
        return sorted(results, key=lambda x: x['p_value'])

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
