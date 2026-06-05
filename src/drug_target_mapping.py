"""
Drug Target Mapping & Druggability Assessment

Integrate DrugBank and ChEMBL to identify FDA-approved drugs and clinical
candidates targeting the top biomarker genes. Classify therapeutic targets
by druggability tier (Tier 1: FDA-approved → Tier 4: Preclinical).

Author: TARVIA-lab
"""

import pandas as pd
import numpy as np
import requests
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DrugInfo:
    """Drug target information"""
    drug_name: str
    gene_symbol: str
    drug_id: str
    source: str  # 'DrugBank', 'ChEMBL', 'FDA'
    tier: str  # 'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'
    indication: str
    mechanism: str
    status: str  # 'FDA-approved', 'Clinical Trial', 'Preclinical'
    synonyms: List[str]


class DrugTargetMapper:
    """
    Map biomarker genes to known drugs and assess druggability.

    Tiers:
    - Tier 1: FDA-approved therapies
    - Tier 2: Drugs in clinical trials (Phase 2+)
    - Tier 3: Preclinical evidence or investigational compounds
    - Tier 4: No known targeting compounds
    """

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # In-memory drug database (can be pre-populated or loaded from API)
        self.drug_db = self._initialize_drug_database()

        # Gene-drug mapping
        self.gene_drug_map = defaultdict(list)

    def _initialize_drug_database(self) -> Dict:
        """
        Initialize curated drug database with common cancer targets.

        In production, this would load from:
        - DrugBank XML dump (requires free academic account)
        - ChEMBL REST API
        - FDA approval database

        For MVP, use manually curated list of common oncology drugs.
        """
        # Common ovarian cancer drugs and targets
        drugs = {
            'EGFR': [
                {'name': 'Erlotinib', 'tier': 'Tier 1', 'status': 'FDA-approved',
                 'indication': 'NSCLC with EGFR mutation', 'mechanism': 'TKI'},
                {'name': 'Gefitinib', 'tier': 'Tier 1', 'status': 'FDA-approved',
                 'indication': 'NSCLC with EGFR mutation', 'mechanism': 'TKI'},
                {'name': 'Afatinib', 'tier': 'Tier 1', 'status': 'FDA-approved',
                 'indication': 'NSCLC with EGFR mutation', 'mechanism': 'TKI'},
            ],
            'TP53': [
                {'name': 'APR-246', 'tier': 'Tier 2', 'status': 'Clinical Trial',
                 'indication': 'TP53 mutant cancers', 'mechanism': 'p53 reactivator'},
                {'name': 'COTI-2', 'tier': 'Tier 3', 'status': 'Clinical Trial',
                 'indication': 'TP53 mutant tumors', 'mechanism': 'p53 restoration'},
            ],
            'BRCA1': [
                {'name': 'Olaparib', 'tier': 'Tier 1', 'status': 'FDA-approved',
                 'indication': 'BRCA1/2 mutant ovarian/breast cancer', 'mechanism': 'PARP inhibitor'},
                {'name': 'Niraparib', 'tier': 'Tier 1', 'status': 'FDA-approved',
                 'indication': 'BRCA1/2 mutant cancers', 'mechanism': 'PARP inhibitor'},
                {'name': 'Rucaparib', 'tier': 'Tier 1', 'status': 'FDA-approved',
                 'indication': 'BRCA1/2 mutant ovarian cancer', 'mechanism': 'PARP inhibitor'},
            ],
            'BRCA2': [
                {'name': 'Olaparib', 'tier': 'Tier 1', 'status': 'FDA-approved',
                 'indication': 'BRCA1/2 mutant ovarian/breast cancer', 'mechanism': 'PARP inhibitor'},
                {'name': 'Niraparib', 'tier': 'Tier 1', 'status': 'FDA-approved',
                 'indication': 'BRCA1/2 mutant cancers', 'mechanism': 'PARP inhibitor'},
            ],
            'PIK3CA': [
                {'name': 'Alpelisib', 'tier': 'Tier 1', 'status': 'FDA-approved',
                 'indication': 'HR+/HER2- breast cancer with PIK3CA mutation', 'mechanism': 'PI3K inhibitor'},
                {'name': 'Taselisib', 'tier': 'Tier 2', 'status': 'Clinical Trial',
                 'indication': 'Advanced cancer with PIK3CA mutation', 'mechanism': 'PI3K inhibitor'},
            ],
            'KRAS': [
                {'name': 'Sotorasib', 'tier': 'Tier 1', 'status': 'FDA-approved',
                 'indication': 'KRAS G12C mutant NSCLC', 'mechanism': 'KRAS inhibitor'},
                {'name': 'Adagrasib', 'tier': 'Tier 1', 'status': 'FDA-approved',
                 'indication': 'KRAS G12C mutant colorectal cancer', 'mechanism': 'KRAS inhibitor'},
            ],
            'BRAF': [
                {'name': 'Vemurafenib', 'tier': 'Tier 1', 'status': 'FDA-approved',
                 'indication': 'BRAF V600E melanoma', 'mechanism': 'BRAF inhibitor'},
                {'name': 'Dabrafenib', 'tier': 'Tier 1', 'status': 'FDA-approved',
                 'indication': 'BRAF V600E melanoma', 'mechanism': 'BRAF inhibitor'},
            ],
            'PTEN': [
                {'name': 'Buparlisib', 'tier': 'Tier 2', 'status': 'Clinical Trial',
                 'indication': 'PTEN-deficient cancers', 'mechanism': 'PI3K/mTOR inhibitor'},
                {'name': 'GSK2636771', 'tier': 'Tier 3', 'status': 'Clinical Trial',
                 'indication': 'PTEN-lost tumors', 'mechanism': 'PI3K inhibitor'},
            ],
            'NFKB1': [
                {'name': 'IKK16', 'tier': 'Tier 3', 'status': 'Preclinical',
                 'indication': 'NF-κB driven cancers', 'mechanism': 'IKK inhibitor'},
                {'name': 'MLM603', 'tier': 'Tier 3', 'status': 'Preclinical',
                 'indication': 'NF-κB pathway', 'mechanism': 'RelB antagonist'},
            ],
            'RELB': [
                {'name': 'MLM603', 'tier': 'Tier 3', 'status': 'Clinical Trial',
                 'indication': 'RelB-dependent tumors', 'mechanism': 'RelB inhibitor'},
                {'name': 'RelB inhibitor X', 'tier': 'Tier 4', 'status': 'Preclinical',
                 'indication': 'Experimental', 'mechanism': 'RelB blockade'},
            ],
        }
        return drugs

    def map_genes_to_drugs(self, gene_symbols: List[str]) -> pd.DataFrame:
        """
        Map a list of gene symbols to known drugs.

        Args:
            gene_symbols: List of gene symbols to query

        Returns:
            DataFrame with columns: gene, drug_name, tier, status, indication, mechanism
        """
        results = []

        for gene in gene_symbols:
            if gene in self.drug_db:
                for drug_info in self.drug_db[gene]:
                    results.append({
                        'gene': gene,
                        'drug_name': drug_info['name'],
                        'tier': drug_info['tier'],
                        'status': drug_info['status'],
                        'indication': drug_info['indication'],
                        'mechanism': drug_info['mechanism'],
                    })
            else:
                # Gene with no known drugs
                results.append({
                    'gene': gene,
                    'drug_name': None,
                    'tier': 'Tier 4',
                    'status': 'No known targeting',
                    'indication': 'Unknown',
                    'mechanism': 'Unknown',
                })

        df = pd.DataFrame(results)
        return df

    def assess_druggability(self, genes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assess druggability for each gene based on:
        - Tier 1 drugs available (score: 4)
        - Tier 2 drugs available (score: 3)
        - Tier 3 drugs available (score: 2)
        - No known drugs (score: 0)

        Args:
            genes_df: DataFrame with 'gene' column

        Returns:
            DataFrame with druggability scores and recommendations
        """
        druggability = []

        for gene in genes_df['gene']:
            if gene in self.drug_db:
                drugs = self.drug_db[gene]
                tiers = [d['tier'] for d in drugs]

                if 'Tier 1' in tiers:
                    score = 4
                    best_tier = 'Tier 1 - FDA-approved'
                elif 'Tier 2' in tiers:
                    score = 3
                    best_tier = 'Tier 2 - Clinical Trial'
                elif 'Tier 3' in tiers:
                    score = 2
                    best_tier = 'Tier 3 - Preclinical'
                else:
                    score = 1
                    best_tier = 'Tier 4 - No targeting'

                n_drugs = len(drugs)

            else:
                score = 0
                best_tier = 'Tier 4 - Undruggable'
                n_drugs = 0

            druggability.append({
                'gene': gene,
                'druggability_score': score,
                'best_tier': best_tier,
                'n_drugs': n_drugs,
                'recommendation': self._get_recommendation(score, gene),
            })

        return pd.DataFrame(druggability)

    def _get_recommendation(self, score: int, gene: str) -> str:
        """Generate therapeutic recommendation based on druggability score."""
        if score == 4:
            return 'Immediate clinical application - FDA-approved drugs available'
        elif score == 3:
            return 'Near-term clinical utility - drugs in clinical trials'
        elif score == 2:
            return 'Research focus - preclinical compounds, requires development'
        elif score == 1:
            return 'Exploratory target - limited drugging evidence'
        else:
            return 'Undruggable with current approaches - requires novel therapeutics'

    def query_drugbank_api(self, gene_symbol: str,
                          drugbank_api_key: Optional[str] = None) -> List[Dict]:
        """
        Query DrugBank REST API for gene-drug associations.

        Requires free academic DrugBank API key from:
        https://www.drugbank.ca/

        Args:
            gene_symbol: Gene symbol to query
            drugbank_api_key: DrugBank API key (or set DRUGBANK_API_KEY env var)

        Returns:
            List of drug dictionaries
        """
        if drugbank_api_key is None:
            import os
            drugbank_api_key = os.environ.get('DRUGBANK_API_KEY')

        if not drugbank_api_key:
            logger.warning('DrugBank API key not provided. Returning curated database only.')
            return []

        url = f'https://www.drugbank.ca/api/v1/drugs'
        params = {
            'q': f'targets:{gene_symbol}',
            'type': 'exact'
        }
        headers = {'Authorization': f'Bearer {drugbank_api_key}'}

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json().get('drugs', [])
        except requests.RequestException as e:
            logger.error(f'DrugBank API error for {gene_symbol}: {e}')
            return []

    def query_chembl_api(self, gene_symbol: str) -> List[Dict]:
        """
        Query ChEMBL API for compound information targeting a gene.

        ChEMBL API is free and doesn't require authentication.
        Endpoint: https://www.ebi.ac.uk/chembl/api/data/

        Args:
            gene_symbol: Gene symbol to query

        Returns:
            List of compound dictionaries
        """
        url = f'https://www.ebi.ac.uk/chembl/api/data/targets'
        params = {
            'target_synonym__icontains': gene_symbol,
            'format': 'json',
            'limit': 100
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            targets = response.json().get('results', [])

            compounds = []
            for target in targets:
                target_id = target.get('target_chembl_id')
                if target_id:
                    compounds.extend(self._get_compounds_for_target(target_id))

            return compounds
        except requests.RequestException as e:
            logger.error(f'ChEMBL API error for {gene_symbol}: {e}')
            return []

    def _get_compounds_for_target(self, chembl_target_id: str) -> List[Dict]:
        """Get compounds with bioassay data for a ChEMBL target ID."""
        url = f'https://www.ebi.ac.uk/chembl/api/data/activities'
        params = {
            'target_chembl_id__exact': chembl_target_id,
            'format': 'json',
            'limit': 50,
            'ordering': '-potency'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('results', [])
        except requests.RequestException as e:
            logger.error(f'ChEMBL bioassay error for {chembl_target_id}: {e}')
            return []

    def generate_drug_target_report(self, genes_df: pd.DataFrame,
                                    cox_results: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate comprehensive drug-target report.

        Args:
            genes_df: DataFrame with 'gene' column (e.g., top 30 biomarkers)
            cox_results: Optional DataFrame with 'gene', 'hazard_ratio', 'p_value'

        Returns:
            Merged DataFrame with gene info, Cox results, drugs, and druggability
        """
        # Map genes to drugs
        drug_mapping = self.map_genes_to_drugs(genes_df['gene'].tolist())

        # Assess druggability
        druggability = self.assess_druggability(genes_df)

        # Merge with Cox results if provided
        if cox_results is not None:
            report = genes_df.merge(cox_results[['gene', 'hazard_ratio', 'log2_HR', 'p_value']],
                                   on='gene', how='left')
        else:
            report = genes_df.copy()

        # Add drug mapping (group by gene)
        drug_summary = drug_mapping.groupby('gene').agg({
            'drug_name': lambda x: '; '.join(x.dropna().unique()),
            'tier': 'first',
            'mechanism': lambda x: '; '.join(x.dropna().unique()),
        }).reset_index()

        # Add druggability scores
        report = report.merge(druggability, on='gene', how='left')
        report = report.merge(drug_summary, on='gene', how='left')

        # Sort by druggability score (highest first)
        report = report.sort_values('druggability_score', ascending=False)

        return report

    def save_drug_target_report(self, report: pd.DataFrame,
                               filename: str = 'drug_target_mapping.csv') -> Path:
        """Save drug-target report to CSV."""
        filepath = self.results_dir / filename
        report.to_csv(filepath, index=False)
        logger.info(f'Saved drug-target report to {filepath}')
        return filepath

    def export_for_visualization(self, report: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Export data in formats suitable for network/bubble plot visualization.

        Returns:
            Dictionary with:
            - 'nodes': Gene and drug nodes for network plot
            - 'edges': Gene-drug connections with druggability tier
            - 'bubble': Summary for bubble plot (gene, druggability, n_drugs, pathway)
        """
        # Nodes: genes and drugs
        gene_nodes = pd.DataFrame({
            'id': report['gene'],
            'type': 'gene',
            'druggability_score': report['druggability_score'],
        })

        # Extract unique drugs
        all_drugs = []
        for _, row in report.iterrows():
            if pd.notna(row['drug_name']):
                drugs = str(row['drug_name']).split('; ')
                all_drugs.extend(drugs)

        drug_nodes = pd.DataFrame({
            'id': list(set(all_drugs)),
            'type': 'drug',
        })

        # Edges: gene-drug connections
        edges = []
        for _, row in report.iterrows():
            if pd.notna(row['drug_name']):
                drugs = str(row['drug_name']).split('; ')
                for drug in drugs:
                    edges.append({
                        'source': row['gene'],
                        'target': drug,
                        'tier': row['tier'],
                        'druggability_score': row['druggability_score'],
                    })

        edges_df = pd.DataFrame(edges) if edges else pd.DataFrame()

        # Bubble plot data
        bubble_data = report[[
            'gene', 'druggability_score', 'n_drugs', 'best_tier', 'recommendation'
        ]].copy()

        return {
            'gene_nodes': gene_nodes,
            'drug_nodes': drug_nodes,
            'edges': edges_df,
            'bubble': bubble_data,
        }


def main():
    """Example usage of DrugTargetMapper."""
    mapper = DrugTargetMapper()

    # Example: Map top genes
    top_genes = pd.DataFrame({
        'gene': ['EGFR', 'TP53', 'BRCA1', 'BRCA2', 'PIK3CA', 'KRAS', 'PTEN', 'RELB']
    })

    # Generate report
    report = mapper.generate_drug_target_report(top_genes)
    mapper.save_drug_target_report(report)

    # Export for visualization
    viz_data = mapper.export_for_visualization(report)

    print(f'\n✓ Drug-target mapping complete')
    print(f'  - {len(report)} genes mapped')
    print(f'  - Druggable targets (Tier 1-3): {(report["druggability_score"] > 0).sum()}')
    print(f'  - Undruggable (Tier 4): {(report["druggability_score"] == 0).sum()}')

    return report


if __name__ == '__main__':
    main()
