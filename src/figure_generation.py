"""
Publication-Ready Figure Generation

Create high-quality figures for manuscript submission:
- 5 main figures for primary results
- 3 supplementary figures for supporting evidence

All figures are generated at 300 DPI with Nature/Science styling guidelines.

Author: TARVIA-lab
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# Optional visualization libraries
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from matplotlib.patches import Rectangle
    import matplotlib.patches as patches
    MATPLOTLIB_PATCHES_AVAILABLE = True
except ImportError:
    MATPLOTLIB_PATCHES_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Publication-ready style settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.constrained_layout.use'] = True

sns.set_style('whitegrid', {'grid.alpha': 0.3})
sns.set_palette('husl')


class FigureGenerator:
    """
    Generate publication-ready figures for biomarker discovery pipeline.

    Figures follow Nature/Science submission guidelines:
    - 300 DPI resolution
    - 1.5-2 column width
    - Clear legends and captions
    - Color-blind friendly palettes
    - Consistent fonts and spacing
    """

    def __init__(self, data_dir: str = 'data', output_format: str = 'png'):
        """
        Initialize FigureGenerator.

        Args:
            data_dir: Root data directory
            output_format: Output format ('png', 'pdf', 'svg')
        """
        self.data_dir = Path(data_dir)
        self.figures_dir = self.data_dir.parent / 'figures'
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.output_format = output_format
        self.dpi = 300

        # Color palettes
        self.palette_blue_red = sns.color_palette('RdBu_r', 256)
        self.palette_volcano = sns.diverging_palette(250, 10, as_cmap=True)

    def figure_1_signature_discovery(self, cox_results: pd.DataFrame,
                                     expression_matrix: Optional[pd.DataFrame] = None,
                                     top_n: int = 30) -> Path:
        """
        Figure 1: Signature Discovery

        Panel A: Volcano plot of Cox regression results
        Panel B: Heatmap of top genes across datasets

        Args:
            cox_results: DataFrame with 'gene', 'log2_HR', 'p_value'
            expression_matrix: Optional expression matrix for heatmap (genes x samples)
            top_n: Number of top genes to show in heatmap

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Panel A: Volcano plot
        ax_a = axes[0]
        log2hr = cox_results['log2_HR'].values
        neg_log10_p = -np.log10(cox_results['p_value'].values)

        # Color points by significance
        colors = ['red' if (abs(lhr) > 1 and p < 0.05) else 'gray'
                 for lhr, p in zip(log2hr, neg_log10_p)]

        ax_a.scatter(log2hr, neg_log10_p, alpha=0.6, s=30, c=colors, edgecolors='none')

        # Add threshold lines
        ax_a.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_a.axvline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_a.axvline(-1, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Annotate top genes
        top_genes = cox_results.nlargest(5, 'log2_HR')
        for _, gene in top_genes.iterrows():
            ax_a.annotate(gene['gene'],
                         xy=(gene['log2_HR'], -np.log10(gene['p_value'])),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax_a.set_xlabel('Log₂(Hazard Ratio)', fontsize=10)
        ax_a.set_ylabel('-Log₁₀(p-value)', fontsize=10)
        ax_a.set_title('A. Cox Regression: Prognostic Genes', fontsize=11, fontweight='bold')
        ax_a.grid(True, alpha=0.3)

        # Panel B: Heatmap (placeholder if no expression data)
        ax_b = axes[1]
        if expression_matrix is not None:
            # Select top genes
            top_genes_list = cox_results.nlargest(top_n, 'log2_HR')['gene'].tolist()
            if all(g in expression_matrix.index for g in top_genes_list):
                heatmap_data = expression_matrix.loc[top_genes_list]

                # Standardize each row
                heatmap_data = (heatmap_data - heatmap_data.mean(axis=1).values[:, None]) / \
                              heatmap_data.std(axis=1).values[:, None]

                # Create heatmap
                sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, ax=ax_b,
                           cbar_kws={'label': 'Standardized Expression'},
                           xticklabels=False, yticklabels=True)
                ax_b.set_title(f'B. Expression: Top {top_n} Genes', fontsize=11, fontweight='bold')
            else:
                ax_b.text(0.5, 0.5, f'Heatmap: Top {top_n} genes\n(expression data not provided)',
                         ha='center', va='center', transform=ax_b.transAxes)
        else:
            ax_b.text(0.5, 0.5, f'Heatmap: Top {top_n} genes\n(expression data not provided)',
                     ha='center', va='center', transform=ax_b.transAxes)

        plt.suptitle('Figure 1: Biomarker Signature Discovery', fontsize=12, fontweight='bold', y=1.02)
        filepath = self.figures_dir / f'Figure_1.{self.output_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f'Saved Figure 1 to {filepath}')
        plt.close()

        return filepath

    def figure_2_forest_plot(self, cox_results: pd.DataFrame,
                            top_n: int = 30) -> Path:
        """
        Figure 2: Forest Plot

        Hazard ratios with 95% confidence intervals for top prognostic genes.

        Args:
            cox_results: DataFrame with 'gene', 'hazard_ratio', 'ci_lower', 'ci_upper'
            top_n: Number of top genes to display

        Returns:
            Path to saved figure
        """
        # Select top genes
        top_genes = cox_results.nlargest(top_n, 'hazard_ratio')[::-1]

        fig, ax = plt.subplots(figsize=(10, 10))

        y_pos = np.arange(len(top_genes))

        # Extract data
        hrs = top_genes['hazard_ratio'].values
        ci_lower = top_genes['ci_lower'].values
        ci_upper = top_genes['ci_upper'].values
        genes = top_genes['gene'].values

        # Plot confidence intervals
        errors = [hrs - ci_lower, ci_upper - hrs]
        ax.errorbar(hrs, y_pos, xerr=errors, fmt='D', markersize=6,
                   capsize=4, elinewidth=1.5, color='steelblue', zorder=3)

        # Reference line at HR=1
        ax.axvline(1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='HR=1 (null)')

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(genes)
        ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=11, fontweight='bold')
        ax.set_title(f'Figure 2: Forest Plot of Top {top_n} Prognostic Genes',
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='lower right')

        filepath = self.figures_dir / f'Figure_2.{self.output_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f'Saved Figure 2 to {filepath}')
        plt.close()

        return filepath

    def figure_3_pathway_enrichment(self, pathway_results: pd.DataFrame) -> Path:
        """
        Figure 3: Pathway Enrichment Bubble Plot

        Bubble plot showing NES vs -log10(p-value), colored by pathway direction.

        Args:
            pathway_results: DataFrame with 'pathway', 'nes', 'pval', 'gene_count'

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Filter significant pathways
        pathway_sig = pathway_results[pathway_results['pval'] < 0.05].copy()

        if len(pathway_sig) == 0:
            logger.warning('No significant pathways found')
            return None

        # Create bubble plot
        scatter = ax.scatter(pathway_sig['nes'],
                           -np.log10(pathway_sig['pval']),
                           s=pathway_sig['gene_count'] * 5,
                           c=pathway_sig['nes'],
                           cmap='RdBu_r',
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)

        # Annotate top pathways
        for idx, row in pathway_sig.nlargest(10, 'pval').iterrows():
            ax.annotate(row['pathway'][:40],
                       xy=(row['nes'], -np.log10(row['pval'])),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

        # Reference lines
        ax.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        ax.set_xlabel('Normalized Enrichment Score (NES)', fontsize=11, fontweight='bold')
        ax.set_ylabel('-Log₁₀(p-value)', fontsize=11, fontweight='bold')
        ax.set_title('Figure 3: Pathway Enrichment Analysis',
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('NES', fontsize=10)

        filepath = self.figures_dir / f'Figure_3.{self.output_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f'Saved Figure 3 to {filepath}')
        plt.close()

        return filepath

    def figure_4_kaplan_meier(self, survival_data: pd.DataFrame,
                              gene_symbols: List[str],
                              time_col: str = 'os_months',
                              event_col: str = 'os_event') -> Path:
        """
        Figure 4: Kaplan-Meier Survival Curves

        Plot K-M curves for top genes, stratified by median expression.

        Args:
            survival_data: DataFrame with gene expression + survival columns
            gene_symbols: List of gene symbols to plot (top 4-6 recommended)
            time_col: Column name for survival time
            event_col: Column name for event indicator

        Returns:
            Path to saved figure
        """
        n_genes = min(len(gene_symbols), 6)
        n_cols = 3
        n_rows = (n_genes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_genes > 1 else [axes]

        from lifelines import KaplanMeierFitter

        for idx, gene in enumerate(gene_symbols[:n_genes]):
            ax = axes[idx]

            if gene not in survival_data.columns:
                ax.text(0.5, 0.5, f'{gene} not in data', ha='center', va='center',
                       transform=ax.transAxes)
                continue

            # Stratify by median
            median_expr = survival_data[gene].median()
            high = survival_data[survival_data[gene] > median_expr]
            low = survival_data[survival_data[gene] <= median_expr]

            kmf = KaplanMeierFitter()

            # High expression
            kmf.fit(high[time_col], high[event_col], label='High')
            kmf.plot_survival_function(ax=ax, ci_show=True, color='red', linewidth=2)

            # Low expression
            kmf.fit(low[time_col], low[event_col], label='Low')
            kmf.plot_survival_function(ax=ax, ci_show=True, color='blue', linewidth=2)

            ax.set_xlabel('Time (months)', fontsize=10)
            ax.set_ylabel('Overall Survival', fontsize=10)
            ax.set_title(f'{gene}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower left')

        # Hide empty subplots
        for idx in range(n_genes, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Figure 4: Kaplan-Meier Survival Curves',
                    fontsize=12, fontweight='bold', y=1.00)

        filepath = self.figures_dir / f'Figure_4.{self.output_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f'Saved Figure 4 to {filepath}')
        plt.close()

        return filepath

    def figure_5_drug_gene_network(self, drug_gene_df: pd.DataFrame,
                                   top_n: int = 30) -> Path:
        """
        Figure 5: Drug-Gene Interaction Network

        Network visualization of top genes and associated drugs.

        Args:
            drug_gene_df: DataFrame with 'gene', 'drug_name', 'tier'
            top_n: Number of top genes to include

        Returns:
            Path to saved figure
        """
        if not NETWORKX_AVAILABLE:
            logger.warning('NetworkX not available. Skipping Figure 5.')
            return None

        fig, ax = plt.subplots(figsize=(14, 12))

        # Create graph
        G = nx.Graph()

        # Add nodes and edges
        for _, row in drug_gene_df.head(top_n).iterrows():
            gene = row['gene']
            if pd.notna(row['drug_name']):
                drugs = str(row['drug_name']).split('; ')
                for drug in drugs:
                    tier = row['tier'] if 'tier' in row.index else 'Unknown'
                    G.add_edge(gene, drug, tier=tier)

        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw nodes
        gene_nodes = [n for n in G.nodes() if n in drug_gene_df['gene'].values]
        drug_nodes = [n for n in G.nodes() if n not in gene_nodes]

        nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, node_color='lightblue',
                              node_size=800, label='Genes', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color='lightcoral',
                              node_size=600, label='Drugs', ax=ax)

        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax)

        ax.set_title('Figure 5: Drug-Gene Interaction Network',
                    fontsize=12, fontweight='bold', pad=15)
        ax.axis('off')
        ax.legend(loc='upper left', fontsize=10)

        filepath = self.figures_dir / f'Figure_5.{self.output_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f'Saved Figure 5 to {filepath}')
        plt.close()

        return filepath

    def supp_figure_1_batch_correction(self, expr_before: pd.DataFrame,
                                       expr_after: pd.DataFrame,
                                       batch_labels: pd.Series) -> Path:
        """
        Supplementary Figure 1: Batch Correction Quality

        PCA plots before/after ComBat batch correction.

        Args:
            expr_before: Expression matrix before correction
            expr_after: Expression matrix after correction
            batch_labels: Batch assignment for each sample

        Returns:
            Path to saved figure
        """
        from sklearn.decomposition import PCA

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PCA before
        pca = PCA(n_components=2)
        pcs_before = pca.fit_transform(expr_before.T)
        colors = pd.Categorical(batch_labels).codes

        scatter1 = axes[0].scatter(pcs_before[:, 0], pcs_before[:, 1], c=colors, cmap='tab10',
                                  alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        axes[0].set_title('Before ComBat Correction', fontsize=11, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # PCA after
        pca = PCA(n_components=2)
        pcs_after = pca.fit_transform(expr_after.T)

        scatter2 = axes[1].scatter(pcs_after[:, 0], pcs_after[:, 1], c=colors, cmap='tab10',
                                  alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        axes[1].set_title('After ComBat Correction', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Supplementary Figure 1: Batch Correction Quality',
                    fontsize=12, fontweight='bold', y=1.02)

        filepath = self.figures_dir / f'Supplementary_Figure_1.{self.output_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f'Saved Supplementary Figure 1 to {filepath}')
        plt.close()

        return filepath

    def supp_figure_2_bootstrap_ci(self, bootstrap_results: pd.DataFrame,
                                   top_n: int = 10) -> Path:
        """
        Supplementary Figure 2: Bootstrap Stability

        95% confidence intervals for hazard ratios across 1000 bootstrap resamples.

        Args:
            bootstrap_results: DataFrame with 'gene', 'hr_mean', 'hr_ci_lower', 'hr_ci_upper'
            top_n: Number of top genes to display

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        top_genes = bootstrap_results.nlargest(top_n, 'hr_mean')[::-1]

        y_pos = np.arange(len(top_genes))

        hrs = top_genes['hr_mean'].values
        ci_lower = top_genes['hr_ci_lower'].values
        ci_upper = top_genes['hr_ci_upper'].values
        genes = top_genes['gene'].values

        errors = [hrs - ci_lower, ci_upper - hrs]
        ax.errorbar(hrs, y_pos, xerr=errors, fmt='o', markersize=8,
                   capsize=5, elinewidth=2, color='darkgreen', zorder=3)

        ax.axvline(1, color='red', linestyle='--', linewidth=2, alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(genes)
        ax.set_xlabel('Hazard Ratio (95% Bootstrap CI)', fontsize=11, fontweight='bold')
        ax.set_title('Supplementary Figure 2: Bootstrap Stability (n=1000)',
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')

        filepath = self.figures_dir / f'Supplementary_Figure_2.{self.output_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f'Saved Supplementary Figure 2 to {filepath}')
        plt.close()

        return filepath

    def supp_figure_3_ml_roc_curves(self, fpr_dict: Dict[str, np.ndarray],
                                    tpr_dict: Dict[str, np.ndarray],
                                    auc_dict: Dict[str, float]) -> Path:
        """
        Supplementary Figure 3: ML Classification ROC Curves

        ROC curves for platinum response and RelB-inhibitor response prediction.

        Args:
            fpr_dict: Dictionary of model_name -> false positive rates
            tpr_dict: Dictionary of model_name -> true positive rates
            auc_dict: Dictionary of model_name -> AUC scores

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot ROC curves
        colors = sns.color_palette('husl', len(fpr_dict))
        for (model, fpr), (_, tpr), (_, auc), color in zip(
            fpr_dict.items(), tpr_dict.items(), auc_dict.items(), colors):

            ax.plot(fpr, tpr, label=f'{model} (AUC={auc:.3f})',
                   linewidth=2.5, color=color)

        # Reference line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random classifier')

        ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax.set_title('Supplementary Figure 3: ML Classification Performance',
                    fontsize=12, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        filepath = self.figures_dir / f'Supplementary_Figure_3.{self.output_format}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f'Saved Supplementary Figure 3 to {filepath}')
        plt.close()

        return filepath

    def generate_all_figures(self, cox_results: pd.DataFrame,
                            pathway_results: pd.DataFrame,
                            drug_gene_results: pd.DataFrame,
                            **kwargs) -> Dict[str, Path]:
        """
        Generate all main and supplementary figures.

        Args:
            cox_results: Cox regression results
            pathway_results: Pathway enrichment results
            drug_gene_results: Drug-gene mapping results
            **kwargs: Additional arguments for specific figures

        Returns:
            Dictionary of figure_name -> filepath
        """
        figures = {}

        logger.info('Generating main figures...')

        # Main figures
        figures['Figure_1'] = self.figure_1_signature_discovery(cox_results, **kwargs)
        figures['Figure_2'] = self.figure_2_forest_plot(cox_results)
        figures['Figure_3'] = self.figure_3_pathway_enrichment(pathway_results)
        if 'survival_data' in kwargs:
            figures['Figure_4'] = self.figure_4_kaplan_meier(**kwargs)
        figures['Figure_5'] = self.figure_5_drug_gene_network(drug_gene_results)

        logger.info('Generating supplementary figures...')

        # Supplementary figures (if data available)
        if 'expr_before' in kwargs and 'expr_after' in kwargs:
            figures['Supp_Figure_1'] = self.supp_figure_1_batch_correction(**kwargs)

        if 'bootstrap_results' in kwargs:
            figures['Supp_Figure_2'] = self.supp_figure_2_bootstrap_ci(**kwargs)

        if 'fpr_dict' in kwargs:
            figures['Supp_Figure_3'] = self.supp_figure_3_ml_roc_curves(**kwargs)

        logger.info(f'✓ Generated {len([f for f in figures.values() if f])} figures')
        return figures


def main():
    """Example usage of FigureGenerator."""
    gen = FigureGenerator()
    logger.info('FigureGenerator ready. Call generate_all_figures() with your results.')


if __name__ == '__main__':
    main()
