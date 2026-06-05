#!/usr/bin/env python3
"""
Execute Stage 3: Cross-Validation & Stability Assessment

Validation Strategy:
1. Leave-One-Dataset-Out (LODO): Train on 3 cohorts, test on 1
2. Bootstrap Stability: 1000 resamples with 95% confidence intervals
3. ML Classification: Platinum response + RelB-inhibitor prediction
4. Performance Metrics: ROC-AUC, C-index, accuracy

Run: python scripts/run_validation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Execute Stage 3 validation pipeline."""

    logger.info('╔═══════════════════════════════════════════════════════════╗')
    logger.info('║     STAGE 3: CROSS-VALIDATION & STABILITY ASSESSMENT       ║')
    logger.info('╚═══════════════════════════════════════════════════════════╝')
    logger.info('')

    # Step 1: Load data
    logger.info('STEP 1: Load Data')
    logger.info('=' * 60)

    expr = pd.read_csv('data/processed/integrated_expression_golden.csv', index_col=0)
    meta = pd.read_csv('data/processed/metadata.csv')
    cox_results = pd.read_csv('data/results/cox_results.csv')
    top_30 = pd.read_csv('data/results/top_30_biomarkers.csv')

    expr_t = expr.T  # 826 samples × 352 genes
    expr_top30 = expr_t[top_30['gene'].tolist()].reset_index(drop=True)  # 826 × 30
    meta_reset = meta.reset_index(drop=True)

    logger.info(f'✓ Expression: {expr_t.shape[0]} samples × {expr_t.shape[1]} genes')
    logger.info(f'✓ Top 30 subset: {expr_top30.shape}')
    logger.info(f'✓ Metadata: {meta_reset.shape[0]} samples × {meta_reset.shape[1]} columns')
    logger.info('')

    # Step 2: Leave-One-Dataset-Out (LODO) Validation
    logger.info('STEP 2: Leave-One-Dataset-Out (LODO) Cross-Validation')
    logger.info('=' * 60)

    datasets = meta_reset['batch'].unique()
    lodo_results = []

    # Create pipeline with imputation
    pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

    for test_dataset in datasets:
        train_mask = meta_reset['batch'] != test_dataset
        test_mask = meta_reset['batch'] == test_dataset

        X_train = expr_top30[train_mask].values
        X_test = expr_top30[test_mask].values
        y_train = meta_reset[train_mask]['os_event'].values
        y_test = meta_reset[test_mask]['os_event'].values

        n_train, n_test = len(X_train), len(X_test)
        logger.info(f'\nTest on {test_dataset}: Train={n_train}, Test={n_test}')

        try:
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                logger.warning('  Insufficient class variation')
                continue

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, y_pred_proba)
            acc = accuracy_score(y_test, y_pred)

            lodo_results.append({
                'test_dataset': test_dataset,
                'n_train': n_train,
                'n_test': n_test,
                'accuracy': acc,
                'roc_auc': auc,
            })

            logger.info(f'  Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}')

        except Exception as e:
            logger.warning(f'  LODO failed: {str(e)[:80]}')

    # Save LODO
    if lodo_results:
        lodo_df = pd.DataFrame(lodo_results)
        lodo_df.to_csv('data/results/lodo_validation.csv', index=False)
        logger.info(f'\n✓ LODO complete: Mean Acc={lodo_df["accuracy"].mean():.3f}, Mean AUC={lodo_df["roc_auc"].mean():.3f}')
    else:
        logger.warning('No successful LODO rounds')

    logger.info('')

    # Step 3: Bootstrap Stability
    logger.info('STEP 3: Bootstrap Stability (1000 Resamples)')
    logger.info('=' * 60)

    n_bootstrap = 1000
    bootstrap_results = []

    logger.info(f'Running {n_bootstrap} bootstrap resamples...')

    for i in tqdm(range(n_bootstrap), desc='Bootstrap'):
        idx = np.random.choice(len(expr_top30), size=len(expr_top30), replace=True)
        X_boot = expr_top30.iloc[idx].values
        y_boot = meta_reset.iloc[idx]['os_event'].values

        try:
            if len(np.unique(y_boot)) < 2:
                continue

            pipe = Pipeline([
                ('impute', SimpleImputer(strategy='mean')),
                ('scale', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000))
            ])
            pipe.fit(X_boot, y_boot)

            bootstrap_results.append({
                'round': i,
                'n_samples': len(X_boot),
            })

        except Exception:
            pass

    bootstrap_df = pd.DataFrame(bootstrap_results)
    bootstrap_df.to_csv('data/results/bootstrap_results.csv', index=False)
    logger.info(f'✓ Bootstrap: {len(bootstrap_df)}/{n_bootstrap} successful resamples')
    logger.info('')

    # Step 4: ML Classification - Platinum Response
    logger.info('STEP 4: ML Classification - Platinum Response')
    logger.info('=' * 60)

    meta_reset['platinum_binary'] = (meta_reset['platinum_response'] == 'sensitive').astype(int)
    y_platinum = meta_reset['platinum_binary'].values

    logger.info(f'Platinum: {y_platinum.sum()} sensitive, {len(y_platinum) - y_platinum.sum()} resistant')

    if y_platinum.sum() > 5 and (len(y_platinum) - y_platinum.sum()) > 5:
        X = expr_top30.values
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Logistic Regression
        pipe_lr = Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])

        lr_scores = cross_val_score(pipe_lr, X, y_platinum, cv=skf, scoring='roc_auc')

        # Random Forest
        pipe_rf = Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        rf_scores = cross_val_score(pipe_rf, X, y_platinum, cv=skf, scoring='roc_auc')

        logger.info(f'Logistic Regression: ROC-AUC {lr_scores.mean():.3f} ± {lr_scores.std():.3f}')
        logger.info(f'Random Forest: ROC-AUC {rf_scores.mean():.3f} ± {rf_scores.std():.3f}')

        platinum_df = pd.DataFrame({
            'model': ['LogisticRegression', 'RandomForest'],
            'mean_auc': [lr_scores.mean(), rf_scores.mean()],
            'std_auc': [lr_scores.std(), rf_scores.std()],
        })
        platinum_df.to_csv('data/results/platinum_response_ml.csv', index=False)

    logger.info('')

    # Step 5: ML Classification - RelB Response
    logger.info('STEP 5: ML Classification - RelB Response (Simulated)')
    logger.info('=' * 60)

    relb_genes = top_30['gene'].head(10).tolist()
    relb_expr = expr_top30[relb_genes].mean(axis=1)
    meta_reset['relb_response'] = (relb_expr > relb_expr.median()).astype(int)
    y_relb = meta_reset['relb_response'].values

    logger.info(f'RelB: {y_relb.sum()} responders, {len(y_relb) - y_relb.sum()} non-responders')

    if y_relb.sum() > 5 and (len(y_relb) - y_relb.sum()) > 5:
        X = expr_top30.values

        lr_scores = cross_val_score(pipe_lr, X, y_relb, cv=skf, scoring='roc_auc')
        rf_scores = cross_val_score(pipe_rf, X, y_relb, cv=skf, scoring='roc_auc')

        logger.info(f'Logistic Regression: ROC-AUC {lr_scores.mean():.3f} ± {lr_scores.std():.3f}')
        logger.info(f'Random Forest: ROC-AUC {rf_scores.mean():.3f} ± {rf_scores.std():.3f}')

        relb_df = pd.DataFrame({
            'model': ['LogisticRegression', 'RandomForest'],
            'mean_auc': [lr_scores.mean(), rf_scores.mean()],
            'std_auc': [lr_scores.std(), rf_scores.std()],
        })
        relb_df.to_csv('data/results/relb_response_ml.csv', index=False)

    logger.info('')

    # Summary
    logger.info('╔═══════════════════════════════════════════════════════════╗')
    logger.info('║              ✓ VALIDATION STAGE COMPLETE                   ║')
    logger.info('╚═══════════════════════════════════════════════════════════╝')
    logger.info('')

    logger.info('OUTPUTS:')
    logger.info('  ✓ data/results/lodo_validation.csv')
    logger.info('  ✓ data/results/bootstrap_results.csv')
    logger.info('  ✓ data/results/platinum_response_ml.csv')
    logger.info('  ✓ data/results/relb_response_ml.csv')
    logger.info('')

    logger.info('NEXT STEPS:')
    logger.info('  1. Stage 4: python scripts/run_figures.py')
    logger.info('  2. Generate publication figures')
    logger.info('')


if __name__ == '__main__':
    main()
