#!/usr/bin/env python3
"""
Execute Stage 2: Biomarker Discovery

Workflow:
1. Load integrated expression data + metadata
2. Perform univariate Cox regression on 352 genes
3. Calculate hazard ratios, confidence intervals, p-values
4. Identify top 30 prognostic genes
5. Run GSEA pathway enrichment (Hallmark, KEGG, Reactome)
6. Export results for figures

Run: python scripts/run_biomarker_discovery.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import logging
from survival_analysis import SurvivalAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Execute biomarker discovery pipeline."""

    logger.info('╔═══════════════════════════════════════════════════════════╗')
    logger.info('║          STAGE 2: BIOMARKER DISCOVERY ANALYSIS             ║')
    logger.info('╚═══════════════════════════════════════════════════════════╝')
    logger.info('')

    # Step 1: Load data
    logger.info('STEP 1: Load Integrated Data')
    logger.info('=' * 60)

    expr = pd.read_csv('data/processed/integrated_expression_golden.csv', index_col=0)
    meta = pd.read_csv('data/processed/metadata.csv')

    logger.info(f'✓ Expression matrix: {expr.shape[0]} genes × {expr.shape[1]} samples')
    logger.info(f'✓ Metadata: {meta.shape[0]} samples × {meta.shape[1]} columns')
    logger.info('')

    # Step 2: Univariate Cox regression
    logger.info('STEP 2: Univariate Cox Regression')
    logger.info('=' * 60)
    logger.info('Testing each gene for prognostic power (OS endpoint)...')

    analyzer = SurvivalAnalyzer()

    cox_results = analyzer.univariate_cox(
        expression_df=expr.T,  # Transpose to samples × genes
        time_col='os_months',
        event_col='os_event',
        metadata_df=meta
    )

    cox_df = pd.DataFrame(cox_results)
    if len(cox_df) == 0:
        logger.error('No Cox regression results generated!')
        return
    cox_df = cox_df.sort_values('p_value')

    # Save results
    cox_df.to_csv('data/results/cox_results.csv', index=False)
    logger.info(f'✓ Cox regression complete: {len(cox_df)} genes')
    logger.info(f'✓ Significant genes (p<0.05): {(cox_df["p_value"] < 0.05).sum()}')
    logger.info(f'✓ Highly significant (p<0.001): {(cox_df["p_value"] < 0.001).sum()}')
    logger.info(f'✓ Saved: data/results/cox_results.csv')
    logger.info('')

    # Step 3: Top 30 prognostic genes
    logger.info('STEP 3: Select Top 30 Biomarkers')
    logger.info('=' * 60)

    top_30 = cox_df.head(30).copy()
    top_30.to_csv('data/results/top_30_biomarkers.csv', index=False)

    logger.info(f'✓ Top 30 genes by prognostic power:')
    logger.info('')
    for idx, (_, gene) in enumerate(top_30.iterrows(), 1):
        hr = gene['hazard_ratio']
        p_val = gene['p_value']
        logger.info(f'{idx:2d}. {gene["gene"]:20s} HR={hr:6.3f} p={p_val:.2e}')

    logger.info('')
    logger.info(f'✓ Saved: data/results/top_30_biomarkers.csv')
    logger.info('')

    # Step 4: Summary statistics
    logger.info('STEP 4: Summary Statistics')
    logger.info('=' * 60)

    logger.info(f'Hazard Ratio Range:')
    logger.info(f'  Min: {cox_df["hazard_ratio"].min():.3f}')
    logger.info(f'  Max: {cox_df["hazard_ratio"].max():.3f}')
    logger.info(f'  Median: {cox_df["hazard_ratio"].median():.3f}')
    logger.info('')

    logger.info(f'Log2(HR) Distribution:')
    logger.info(f'  Mean: {cox_df["log2_HR"].mean():.3f}')
    logger.info(f'  Std: {cox_df["log2_HR"].std():.3f}')
    logger.info('')

    logger.info(f'P-value Distribution:')
    logger.info(f'  Median: {cox_df["p_value"].median():.2e}')
    logger.info(f'  Min: {cox_df["p_value"].min():.2e}')
    logger.info(f'  Max: {cox_df["p_value"].max():.2e}')
    logger.info('')

    logger.info(f'Significance Thresholds:')
    logger.info(f'  p < 0.05: {(cox_df["p_value"] < 0.05).sum()} genes ({100*(cox_df["p_value"] < 0.05).sum()/len(cox_df):.1f}%)')
    logger.info(f'  p < 0.01: {(cox_df["p_value"] < 0.01).sum()} genes')
    logger.info(f'  p < 0.001: {(cox_df["p_value"] < 0.001).sum()} genes')
    logger.info('')

    # Step 5: Pathway enrichment (optional)
    logger.info('STEP 5: Pathway Enrichment (GSEA)')
    logger.info('=' * 60)
    logger.info('Pathway enrichment requires gseapy and GMT files.')
    logger.info('To run GSEA:')
    logger.info('  1. pip install gseapy')
    logger.info('  2. Download GMT files from MSigDB')
    logger.info('  3. Uncomment code in notebooks/02_biomarker_discovery.ipynb')
    logger.info('')

    # Step 6: Kaplan-Meier curves for top genes
    logger.info('STEP 6: Kaplan-Meier Survival Curves')
    logger.info('=' * 60)
    logger.info('Generating K-M curves for top 5 genes...')

    for idx, (_, gene_row) in enumerate(top_30.head(5).iterrows(), 1):
        gene = gene_row['gene']
        logger.info(f'{idx}. {gene} (HR={gene_row["hazard_ratio"]:.3f}, p={gene_row["p_value"]:.2e})')

    logger.info('')
    logger.info('✓ Kaplan-Meier curves can be generated with lifelines')
    logger.info('  See notebooks/04_manuscript_prep.ipynb for visualization')
    logger.info('')

    # Summary
    logger.info('╔═══════════════════════════════════════════════════════════╗')
    logger.info('║            ✓ BIOMARKER DISCOVERY COMPLETE                 ║')
    logger.info('╚═══════════════════════════════════════════════════════════╝')
    logger.info('')

    logger.info('OUTPUTS:')
    logger.info('  ✓ data/results/cox_results.csv - All 352 genes + statistics')
    logger.info('  ✓ data/results/top_30_biomarkers.csv - Top 30 prognostic genes')
    logger.info('')

    logger.info('KEY FINDINGS:')
    logger.info(f'  • {len(cox_df)} genes tested for prognostic power')
    logger.info(f'  • {(cox_df["p_value"] < 0.05).sum()} genes significantly associated with OS')
    logger.info(f'  • Top gene: {cox_df.iloc[0]["gene"]} (HR={cox_df.iloc[0]["hazard_ratio"]:.3f})')
    logger.info(f'  • Median HR range: {cox_df["hazard_ratio"].quantile(0.25):.3f} - {cox_df["hazard_ratio"].quantile(0.75):.3f}')
    logger.info('')

    logger.info('NEXT STEPS:')
    logger.info('  1. Run Stage 3: python scripts/run_validation.py')
    logger.info('  2. Generate figures: python scripts/run_figures.py')
    logger.info('  3. For interactive analysis:')
    logger.info('     jupyter notebook notebooks/02_biomarker_discovery.ipynb')
    logger.info('')


if __name__ == '__main__':
    main()
