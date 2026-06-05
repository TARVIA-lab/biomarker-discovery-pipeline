"""Data integration module for TCGA/GEO datasets"""
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm

class DataIntegrator:
    """Integrate and normalize TCGA/GEO datasets"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True, parents=True)
    
    def load_golden_genes(self, filepath):
        """Load 352 golden RelB-dependent genes"""
        genes_df = pd.read_csv(filepath)
        # Support both 'gene_symbol' (our format) and 'gene_name' (legacy)
        col = 'gene_symbol' if 'gene_symbol' in genes_df.columns else 'gene_name'
        return genes_df[col].tolist()
    
    def normalize_expression(self, expression_matrix):
        """
        Normalize expression matrix:
        - Log2 transform if needed
        - Quantile normalization
        - Mean centering
        """
        # Assume input is already counts or CPM
        expr = expression_matrix.copy()
        
        # Log2 transform
        expr = np.log2(expr + 1)
        
        # Quantile normalization (simple version)
        expr = expr.rank(axis=0, pct=True)
        expr = expr * 4  # Scale to typical range
        
        # Mean center
        expr = expr - expr.mean(axis=0)
        
        return expr
    
    def integrate_datasets(self, datasets_list):
        """
        Integrate multiple datasets with batch correction

        Parameters:
        -----------
        datasets_list : list of dicts
            [{'name': str, 'expr': DataFrame, 'meta': DataFrame}, ...]

        Returns:
        --------
        dict with keys:
            - 'expression': Integrated expression matrix (genes × samples)
            - 'metadata': Merged metadata with batch column
        """
        normalized = []
        metadata = []

        for dataset in datasets_list:
            name = dataset['name']
            expr = dataset['expr']
            meta = dataset['meta']

            # Normalize
            expr_norm = self.normalize_expression(expr)

            # Add batch info to metadata
            meta = meta.copy()
            meta['batch'] = name

            normalized.append(expr_norm)
            metadata.append(meta)

        # Merge expression (genes × samples)
        # Use inner join on genes (rows)
        merged_expr = pd.concat(normalized, axis=1, join='inner')

        # Merge metadata (samples × columns)
        merged_meta = pd.concat(metadata, axis=0, ignore_index=True)

        return {
            'expression': merged_expr,
            'metadata': merged_meta
        }

print("✓ DataIntegrator loaded")
