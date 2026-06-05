"""
Pathway Enrichment Analysis using GSEA

Perform Gene Set Enrichment Analysis (GSEA) using Cox log2 hazard ratios
as ranking metric. Test Hallmark, KEGG, and Reactome pathways for
enrichment in prognostic genes.

Author: TARVIA-lab
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from urllib.parse import quote

try:
    from gseapy import prerank, Msigdb
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    logging.warning('gseapy not installed. Install with: pip install gseapy')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PathwayResult:
    """GSEA pathway enrichment result"""
    pathway: str
    nes: float  # Normalized Enrichment Score
    pval: float
    padj: float  # Adjusted p-value
    fdr: float
    gene_count: int
    direction: str  # 'Up' or 'Down'
    genes: List[str]  # Genes in pathway that are ranked high


class PathwayAnalyzer:
    """
    Gene Set Enrichment Analysis (GSEA) for biomarker pathways.

    Workflow:
    1. Load Cox regression results (log2 hazard ratios)
    2. Rank genes by log2(HR) or p-value
    3. Query GSEA databases (Hallmark, KEGG, Reactome)
    4. Run GSEA prerank algorithm
    5. Filter significant pathways (FDR < 0.05)
    6. Return formatted results and visualizations
    """

    def __init__(self, data_dir: str = 'data', min_gene_set: int = 15, max_gene_set: int = 500):
        """
        Initialize PathwayAnalyzer.

        Args:
            data_dir: Root data directory
            min_gene_set: Minimum genes in pathway (default: 15)
            max_gene_set: Maximum genes in pathway (default: 500)
        """
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.min_gene_set = min_gene_set
        self.max_gene_set = max_gene_set

        # Pathway databases available
        self.pathways = {
            'Hallmark': 'h.all',  # Hallmark gene sets
            'KEGG': 'c2.cp.kegg',  # KEGG canonical pathways
            'Reactome': 'c2.cp.reactome',  # Reactome pathways
            'GO_BP': 'c5.go.bp',  # Gene Ontology: Biological Process
            'GO_MF': 'c5.go.mf',  # Gene Ontology: Molecular Function
        }

        # Local pathway cache (can be pre-populated)
        self.gmt_cache = {}

    def rank_genes(self, cox_results: pd.DataFrame,
                   ranking_metric: str = 'log2_HR') -> pd.Series:
        """
        Rank genes by Cox regression log2(HR) for GSEA prerank.

        Args:
            cox_results: DataFrame with 'gene', 'log2_HR' (or other metric), 'p_value'
            ranking_metric: Column to use for ranking ('log2_HR', 'p_value', etc.)

        Returns:
            Sorted Series of gene -> log2_HR (highest prognostic genes first)
        """
        if ranking_metric not in cox_results.columns:
            raise ValueError(f'Ranking metric "{ranking_metric}" not in Cox results columns')

        # Create ranking: positive log2_HR = higher hazard = poor prognosis
        ranking = cox_results[['gene', ranking_metric]].drop_duplicates('gene')
        ranking = ranking.set_index('gene')[ranking_metric].sort_values(ascending=False)

        logger.info(f'Ranked {len(ranking)} genes by {ranking_metric}')
        return ranking

    def get_gmt_file(self, pathway_db: str) -> Optional[Path]:
        """
        Get or download GMT file for pathway database.

        GMT files are available from Broad Institute:
        https://www.gsea-msigdb.org/gsea/downloads.jsp

        Args:
            pathway_db: Pathway database identifier ('h.all', 'c2.cp.kegg', etc.)

        Returns:
            Path to local GMT file or None if not available
        """
        gmt_dir = self.data_dir / 'pathways'
        gmt_dir.mkdir(parents=True, exist_ok=True)

        gmt_file = gmt_dir / f'{pathway_db}.gmt'

        # Check if already cached locally
        if gmt_file.exists():
            logger.info(f'Using cached GMT: {gmt_file}')
            return gmt_file

        # Try to download from Broad Institute
        logger.info(f'Attempting to download {pathway_db} from Broad Institute...')
        url = self._construct_gmt_url(pathway_db)

        if url:
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                with open(gmt_file, 'w') as f:
                    f.write(response.text)

                logger.info(f'Downloaded GMT to {gmt_file}')
                return gmt_file

            except requests.RequestException as e:
                logger.warning(f'Failed to download {pathway_db}: {e}')
                return None
        else:
            logger.warning(f'Cannot construct URL for {pathway_db}')
            return None

    def _construct_gmt_url(self, pathway_db: str) -> Optional[str]:
        """Construct download URL for GMT file from Broad Institute."""
        # This is a placeholder; actual Broad downloads may require authentication
        # or may be served differently. For production, pre-download GMT files.

        broad_base = 'https://data.broadinstitute.org/gsea-msigdb/msigdb/release'
        mapping = {
            'h.all': 'h.all.v2023.1.Hs.symbols.gmt',
            'c2.cp.kegg': 'c2.cp.kegg.v2023.1.Hs.symbols.gmt',
            'c2.cp.reactome': 'c2.cp.reactome.v2023.1.Hs.symbols.gmt',
        }

        if pathway_db in mapping:
            return f'{broad_base}/{mapping[pathway_db]}'

        return None

    def load_local_gmt(self, gmt_path: Path) -> Dict[str, List[str]]:
        """
        Load local GMT file into memory.

        GMT format:
        <pathway_name> <description> <gene_1> <gene_2> ... <gene_N>

        Args:
            gmt_path: Path to GMT file

        Returns:
            Dictionary: pathway_name -> list of gene symbols
        """
        pathways = {}

        try:
            with open(gmt_path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        pathway_name = parts[0]
                        genes = parts[2:]
                        pathways[pathway_name] = genes

            logger.info(f'Loaded {len(pathways)} pathways from {gmt_path.name}')
            return pathways

        except FileNotFoundError:
            logger.error(f'GMT file not found: {gmt_path}')
            return {}

    def run_gsea_prerank(self, ranked_genes: pd.Series,
                         gmt_file: Path,
                         pathway_name: str = 'pathways',
                         processes: int = 4) -> Optional[pd.DataFrame]:
        """
        Run GSEA prerank algorithm.

        GSEA prerank tests for enrichment of gene sets using a pre-ranked list
        of genes (ranked by log2 hazard ratio).

        Args:
            ranked_genes: Series of gene -> log2_HR (from rank_genes())
            gmt_file: Path to GMT file with pathways
            pathway_name: Name for output directory
            processes: Number of parallel processes

        Returns:
            DataFrame with enrichment results or None if gseapy unavailable
        """
        if not GSEAPY_AVAILABLE:
            logger.error('gseapy not installed. Cannot run GSEA.')
            return None

        outdir = self.results_dir / f'gsea_{pathway_name}'
        outdir.mkdir(parents=True, exist_ok=True)

        logger.info(f'Running GSEA prerank on {len(ranked_genes)} genes...')

        try:
            pre_res = prerank(
                rnk=ranked_genes.to_dict(),
                gene_sets=str(gmt_file),
                outdir=str(outdir),
                min_size=self.min_gene_set,
                max_size=self.max_gene_set,
                permutation_num=1000,
                seed=42,
                processes=processes,
            )

            # Parse results
            if hasattr(pre_res, 'results') and pre_res.results is not None:
                results_df = pre_res.results.copy()
                logger.info(f'✓ GSEA identified {len(results_df)} pathways (p<0.05)')
                return results_df
            else:
                logger.warning('GSEA returned no results')
                return None

        except Exception as e:
            logger.error(f'GSEA error: {e}')
            return None

    def filter_enrichment_results(self, gsea_results: pd.DataFrame,
                                  fdr_threshold: float = 0.05,
                                  nes_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Filter GSEA results by significance and effect size.

        Args:
            gsea_results: DataFrame from GSEA prerank
            fdr_threshold: Maximum FDR q-value (default: 0.05)
            nes_threshold: Minimum absolute NES (optional)

        Returns:
            Filtered DataFrame sorted by NES
        """
        filtered = gsea_results.copy()

        # Filter by FDR
        if 'fdr q-val' in filtered.columns:
            filtered = filtered[filtered['fdr q-val'] <= fdr_threshold]
        elif 'padj' in filtered.columns:
            filtered = filtered[filtered['padj'] <= fdr_threshold]

        logger.info(f'After FDR filter (q < {fdr_threshold}): {len(filtered)} pathways')

        # Filter by NES if specified
        if nes_threshold is not None:
            if 'nes' in filtered.columns:
                filtered = filtered[np.abs(filtered['nes']) >= nes_threshold]
                logger.info(f'After NES filter (|NES| >= {nes_threshold}): {len(filtered)} pathways')

        # Sort by NES
        if 'nes' in filtered.columns:
            filtered = filtered.sort_values('nes', ascending=False)

        return filtered

    def parse_gsea_gene_list(self, gsea_results: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Extract gene members for each enriched pathway.

        GSEA output includes gene symbols in each pathway that are ranked high.

        Args:
            gsea_results: DataFrame from GSEA prerank

        Returns:
            Dictionary: pathway_name -> list of genes in pathway
        """
        pathway_genes = {}

        if 'leading edge genes' in gsea_results.columns:
            for _, row in gsea_results.iterrows():
                pathway = row['Term'] if 'Term' in row.index else row['pathway']
                genes = str(row['leading edge genes']).split(';')
                genes = [g.strip() for g in genes if g.strip()]
                pathway_genes[pathway] = genes

        return pathway_genes

    def summarize_enrichment(self, gsea_results: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for GSEA results.

        Args:
            gsea_results: Filtered DataFrame from GSEA prerank

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_pathways': len(gsea_results),
            'n_up_regulated': (gsea_results['nes'] > 0).sum() if 'nes' in gsea_results.columns else 0,
            'n_down_regulated': (gsea_results['nes'] < 0).sum() if 'nes' in gsea_results.columns else 0,
            'median_nes': gsea_results['nes'].median() if 'nes' in gsea_results.columns else None,
            'median_fdr': (gsea_results['fdr q-val'].median() if 'fdr q-val' in gsea_results.columns
                          else gsea_results['padj'].median() if 'padj' in gsea_results.columns else None),
        }

        return summary

    def generate_pathway_report(self, cox_results: pd.DataFrame,
                               pathway_dbs: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive pathway enrichment report.

        Runs GSEA on multiple pathway databases and consolidates results.

        Args:
            cox_results: Cox regression results DataFrame
            pathway_dbs: List of pathway databases to query (default: ['h.all', 'c2.cp.kegg'])

        Returns:
            Dictionary with results for each pathway database
        """
        if pathway_dbs is None:
            pathway_dbs = ['h.all', 'c2.cp.kegg', 'c2.cp.reactome']

        # Rank genes
        ranked = self.rank_genes(cox_results)

        all_results = {}

        for db in pathway_dbs:
            logger.info(f'\n=== Testing {db} ===')

            # Get GMT file
            gmt_path = self.get_gmt_file(db)
            if not gmt_path:
                logger.warning(f'Skipping {db} - GMT file unavailable')
                continue

            # Run GSEA
            gsea_results = self.run_gsea_prerank(ranked, gmt_path, pathway_name=db)

            if gsea_results is not None:
                # Filter
                filtered = self.filter_enrichment_results(gsea_results)

                if len(filtered) > 0:
                    # Summary
                    summary = self.summarize_enrichment(filtered)
                    logger.info(f'Summary: {summary}')

                    all_results[db] = {
                        'results': filtered,
                        'summary': summary,
                    }
                else:
                    logger.warning(f'No significant pathways in {db}')

        return all_results

    def save_pathway_results(self, results: Dict[str, Dict],
                            filename_prefix: str = 'pathway_enrichment') -> Path:
        """
        Save pathway enrichment results to CSV.

        Args:
            results: Dictionary from generate_pathway_report()
            filename_prefix: Prefix for output filenames

        Returns:
            Path to saved results directory
        """
        for db_name, db_results in results.items():
            filename = f'{filename_prefix}_{db_name}.csv'
            filepath = self.results_dir / filename

            results_df = db_results['results'].copy()

            # Select relevant columns
            keep_cols = [c for c in ['pathway', 'Term', 'nes', 'pval', 'fdr q-val', 'padj',
                                     'tag %', 'list %', 'signal', 'leading edge genes']
                        if c in results_df.columns]
            results_df = results_df[keep_cols]

            results_df.to_csv(filepath, index=False)
            logger.info(f'Saved {len(results_df)} pathways to {filepath}')

        return self.results_dir

    def create_gsea_summary_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create consolidated summary table across all pathway databases.

        Args:
            results: Dictionary from generate_pathway_report()

        Returns:
            DataFrame with top pathways across all DBs
        """
        all_pathways = []

        for db_name, db_results in results.items():
            results_df = db_results['results']

            for _, row in results_df.iterrows():
                pathway = row['Term'] if 'Term' in row.index else row['pathway']
                all_pathways.append({
                    'pathway': pathway,
                    'database': db_name,
                    'nes': row['nes'] if 'nes' in row.index else None,
                    'pval': row['pval'] if 'pval' in row.index else row['pval'] if 'p-val' in row.index else None,
                    'fdr': (row['fdr q-val'] if 'fdr q-val' in row.index
                            else row['padj'] if 'padj' in row.index else None),
                })

        summary_df = pd.DataFrame(all_pathways)
        summary_df = summary_df.sort_values('nes', ascending=False)

        return summary_df


def main():
    """Example usage of PathwayAnalyzer."""
    analyzer = PathwayAnalyzer()

    # Example: Load Cox results
    # cox_results = pd.read_csv('data/results/cox_results.csv')
    # results = analyzer.generate_pathway_report(cox_results)
    # analyzer.save_pathway_results(results)

    print('PathwayAnalyzer ready. Call generate_pathway_report() with Cox results.')


if __name__ == '__main__':
    main()
