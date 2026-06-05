# TARVIA Biomarker Discovery Pipeline

**AI-driven discovery and validation of RelB-dependent therapeutic targets in HGSOC**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![TARVIA-lab](https://img.shields.io/badge/TARVIA--lab-GitHub-black?logo=github)](https://github.com/TARVIA-lab)

---

## Overview

This pipeline identifies and validates new therapeutic targets from your 352 golden RelB-dependent genes using:

- **TCGA OV** (n=316 HGSOC samples)
- **GEO HGSOC Datasets** (n=500+ samples from GSE32062, GSE9891, GSE26712, GSE51373)
- **Prognostic analysis** (Cox regression, Kaplan-Meier survival)
- **Pathway enrichment** (GSEA across Hallmark, KEGG, Reactome)
- **Drug target mapping** (DrugBank, ChEMBL integration)
- **ML classification** (Predict platinum/RelB-inhibitor response)
- **Cross-validation** (Leave-One-Dataset-Out, bootstrap stability)

## Quick Start

```bash
# Clone
git clone https://github.com/TARVIA-lab/biomarker-discovery-pipeline.git
cd biomarker-discovery-pipeline

# Install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python3 src/run_pipeline.py --golden-genes data/golden_genes.csv
```

## Pipeline Architecture

### Stage 1: Data Integration
- Download TCGA OV RNA-seq (GDC API)
- Download GEO datasets (GEO API)
- Normalize (log2, quantile norm, mean-center)
- Batch correction (ComBat)
- Merge to 816 samples × 352 genes

### Stage 2: Biomarker Discovery
**Analysis A: Prognostic Power**
- Univariate Cox regression (each gene vs overall survival)
- Filter: p < 0.05, |log2-HR| > 0.3
- Rank by p-value & effect size

**Analysis B: Treatment Predictors**
- Stratify: RelB-high vs low (tertile split)
- Test differential expression in platinum-resistant vs sensitive
- Kaplan-Meier curves

**Analysis C: Pathway Enrichment**
- GSEA on ranked genes
- Hallmark, KEGG, Reactome pathways
- Focus: drug resistance, EMT, metabolism

**Analysis D: Drug Target Mapping**
- Cross-reference with DrugBank, ChEMBL
- Assess druggability (known vs novel targets)
- Clinical trial matching

### Stage 3: Cross-Validation
- Leave-One-Dataset-Out (LODO): train on 3, test on 4th
- Bootstrap stability: 1000 resamples, 95% CI
- External validation: independent cohorts

### Stage 4: ML Classification
- Logistic regression: predict platinum response
- Random forest: feature importance ranking
- CV AUC, sensitivity, specificity

## Outputs

### Publication-Ready Figures
- **Figure 1:** Discovery workflow (Venn: 352 → candidates → top 30)
- **Figure 2:** Prognostic power (forest plot, Kaplan-Meier curves)
- **Figure 3:** Clinical utility (RelB stratification, heatmap)
- **Figure 4:** Pathway & drug analysis (network, enrichment)
- **Figure 5:** Cross-validation (bootstrap, LODO concordance)
- **Supp Figs:** All 352 genes ranked, full GSEA results

### Data Files
- `top_30_candidates.csv` — gene, HR, p-value, FDR, pathway, druggable
- `validation_results.csv` — each gene × dataset cross-validation
- `pathway_enrichment.csv` — NES, padj, member genes
- `drug_interactions.csv` — known drugs, clinical trials
- `ml_predictions.csv` — platinum response AUC, feature importance

### Reports
- `RESULTS_SUMMARY.md` — Key findings & clinical implications
- `HTML report` — Interactive candidate profiles with DrugBank/ChEMBL links

## Installation

### Requirements
- Python 3.8+
- ~5 GB disk (for TCGA/GEO data)

### Setup
```bash
# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify
python3 -c "import pandas, lifelines, gseapy; print('✓ All dependencies OK')"
```

## Usage

### Basic Workflow
```python
from src.data_integration import DataIntegrator
from src.survival_analysis import SurvivalAnalyzer
from src.ml_classification import DrugResponsePredictor

# Load golden genes
genes = pd.read_csv('data/golden_genes.csv')

# Integrate datasets
integrator = DataIntegrator()
expr, meta = integrator.integrate_datasets([
    ('TCGA', tcga_expr, tcga_meta),
    ('GSE32062', geo1_expr, geo1_meta),
    # ... more datasets
])

# Prognostic analysis
analyzer = SurvivalAnalyzer()
cox_results = analyzer.univariate_cox(expr, 'OS_time', 'OS_event', meta)

# Predict drug response
predictor = DrugResponsePredictor(model_type='logistic')
predictor.train(expr, meta['platinum_response'])
cv_scores = predictor.cross_validate(expr, meta['platinum_response'])
```

## Project Structure

```
biomarker-discovery-pipeline/
├── src/
│   ├── data_integration.py       # TCGA/GEO download & normalization
│   ├── survival_analysis.py      # Cox regression, K-M curves
│   ├── pathway_analysis.py       # GSEA enrichment
│   ├── drug_target_mapping.py    # DrugBank/ChEMBL integration
│   ├── ml_classification.py      # Platinum/RelB-i response prediction
│   └── figure_generation.py      # Publication-ready plots
├── data/
│   ├── golden_genes.csv          # Your 352 RelB-dependent genes
│   ├── raw/                      # TCGA/GEO downloads (gitignored)
│   ├── processed/                # Normalized, merged data
│   └── results/                  # Analysis outputs
├── notebooks/
│   ├── 01_data_integration.ipynb
│   ├── 02_biomarker_discovery.ipynb
│   ├── 03_validation.ipynb
│   └── 04_manuscript_prep.ipynb
├── figures/                      # Publication-ready PNG (300 DPI)
├── docs/
│   ├── PIPELINE.md              # Technical documentation
│   ├── METHODS.md               # Statistical methods
│   └── RESULTS_SUMMARY.md       # Key findings
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Documentation

- **[PIPELINE.md](docs/PIPELINE.md)** — Complete technical guide
- **[METHODS.md](docs/METHODS.md)** — Statistical methods & validation
- **[RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md)** — Key findings & clinical implications

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.3.0 | Data manipulation |
| numpy | ≥1.20.0 | Numerical computing |
| scipy | ≥1.7.0 | Scientific functions |
| scikit-learn | ≥0.24.0 | ML models & preprocessing |
| lifelines | ≥0.27.0 | Survival analysis |
| gseapy | ≥0.10.8 | GSEA pathway analysis |
| matplotlib | ≥3.4.0 | Plotting |
| seaborn | ≥0.11.0 | Statistical visualization |

## Next Steps

1. **Add golden genes file**: `cp /path/to/golden_genes.csv data/`
2. **Download TCGA/GEO data**: Run `src/data_integration.py`
3. **Run discovery analysis**: Execute `notebooks/02_biomarker_discovery.ipynb`
4. **Generate figures**: Run `src/figure_generation.py`
5. **Prepare manuscript**: See `docs/RESULTS_SUMMARY.md`

## Citation

If you use this pipeline, please cite:

```bibtex
@software{lujano2026biomarker,
  author = {Lujano Olazaba, Omar and others},
  title = {TARVIA Biomarker Discovery Pipeline},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/TARVIA-lab/biomarker-discovery-pipeline}
}
```

## License

MIT License — See [LICENSE](LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

## Contact

- **Questions:** o.lujano13@gmail.com
- **GitHub Issues:** https://github.com/TARVIA-lab/biomarker-discovery-pipeline/issues

---

**Status:** Active development (3-month timeline to publication)
**Last Updated:** 2026-06-05
