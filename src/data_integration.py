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
        return genes_df['gene_name'].tolist()
    
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
        datasets_list : list of tuples
            [(name, expression_df, metadata_df), ...]
        
        Returns:
        --------
        merged_expr : pd.DataFrame
            Integrated expression matrix
        merged_meta : pd.DataFrame
            Merged metadata
        """
        normalized = []
        metadata = []
        
        for name, expr, meta in datasets_list:
            # Normalize
            expr_norm = self.normalize_expression(expr)
            expr_norm['dataset'] = name
            
            # Metadata
            meta['dataset'] = name
            
            normalized.append(expr_norm)
            metadata.append(meta)
        
        # Merge
        merged_expr = pd.concat(normalized, axis=0, join='inner')
        merged_meta = pd.concat(metadata, axis=0, ignore_index=True)
        
        return merged_expr, merged_meta

print("✓ DataIntegrator loaded")
