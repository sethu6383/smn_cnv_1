#!/usr/bin/env python3

"""
control_gene_normalization.py - Enhanced normalization using control genes (CFTR and RNase P)
Usage: python control_gene_normalization.py <coverage_file> <sample_info_file> <control_coverage_file> <output_file>
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Control gene coordinates (GRCh38/hg38)
CONTROL_GENES = {
    'CFTR': {
        'chromosome': 'chr7',
        'start': 117120016,
        'end': 117308718,
        'exons': [
            ('CFTR_exon1', 'chr7', 117120016, 117120140),
            ('CFTR_exon2', 'chr7', 117144363, 117144462),
            ('CFTR_exon3', 'chr7', 117149094, 117149196),
            ('CFTR_exon4', 'chr7', 117170286, 117170412),
            ('CFTR_exon5', 'chr7', 117171052, 117171159)
        ]
    },
    'RPPH1': {  # RNase P
        'chromosome': 'chr14',
        'start': 20649636,
        'end': 20650216,
        'exons': [
            ('RPPH1_exon1', 'chr14', 20649636, 20649736),
            ('RPPH1_exon2', 'chr14', 20649836, 20649936),
            ('RPPH1_exon3', 'chr14', 20650116, 20650216)
        ]
    }
}

def load_control_gene_coverage(control_coverage_file):
    """Load coverage data for control genes."""
    try:
        control_df = pd.read_csv(control_coverage_file, sep='\t')
        print(f"Loaded control gene coverage: {len(control_df)} records")
        return control_df
    except Exception as e:
        print(f"Warning: Could not load control gene coverage ({e}). Will generate placeholder data.")
        return pd.DataFrame()

def read_sample_info(sample_info_file):
    """Read sample information from file."""
    try:
        sample_df = pd.read_csv(sample_info_file, sep='\t')
        samples = {}
        for _, row in sample_df.iterrows():
            samples[row['sample_id']] = {
                'bam_path': row.get('bam_path', ''),
                'sample_type': row.get('sample_type', 'unknown')
            }
        return samples
    except Exception as e:
        print(f"Warning: Could not read sample info file ({e}). Using defaults.")
        return {}

def calculate_control_gene_factors(target_coverage_df, control_coverage_df, reference_samples):
    """Calculate normalization factors based on control genes."""
    normalization_factors = {}
    
    if control_coverage_df.empty:
        print("Warning: No control gene data available. Using target-based normalization only.")
        # Fallback to median normalization using target genes
        for sample_id in target_coverage_df['sample_id'].unique():
            sample_coverage = target_coverage_df[target_coverage_df['sample_id'] == sample_id]['avg_coverage']
            if len(sample_coverage) > 0:
                normalization_factors[sample_id] = {
                    'global_factor': np.median(sample_coverage),
                    'method': 'target_median',
                    'control_genes_available': False
                }
        return normalization_factors
    
    # Calculate reference medians for control genes
    ref_control_medians = {}
    control_exons = control_coverage_df['exon'].unique()
    
    for exon in control_exons:
        ref_values = control_coverage_df[
            (control_coverage_df['exon'] == exon) & 
            (control_coverage_df['sample_id'].isin(reference_samples))
        ]['avg_coverage']
        
        if len(ref_values) > 0:
            ref_control_medians[exon] = np.median(ref_values)
    
    # Calculate per-sample normalization factors
    for sample_id in target_coverage_df['sample_id'].unique():
        sample_control = control_coverage_df[control_coverage_df['sample_id'] == sample_id]
        
        if not sample_control.empty and ref_control_medians:
            # Calculate ratio for each control exon
            ratios = []
            for _, row in sample_control.iterrows():
                exon = row['exon']
                if exon in ref_control_medians and ref_control_medians[exon] > 0:
                    ratio = row['avg_coverage'] / ref_control_medians[exon]
                    if 0.1 < ratio < 10:  # Filter extreme outliers
                        ratios.append(ratio)
            
            if ratios:
                # Use robust statistics
                global_factor = np.median(ratios)
                # Additional factors for batch effects and capture efficiency
                cftr_factor = 1.0
                rnasep_factor = 1.0
                
                cftr_data = sample_control[sample_control['exon'].str.contains('CFTR')]
                if not cftr_data.empty:
                    cftr_median = cftr_data['avg_coverage'].median()
                    cftr_ref_median = np.median([ref_control_medians.get(e, 1) for e in cftr_data['exon']])
                    if cftr_ref_median > 0:
                        cftr_factor = cftr_median / cftr_ref_median
                
                rnasep_data = sample_control[sample_control['exon'].str.contains('RPPH1')]
                if not rnasep_data.empty:
                    rnasep_median = rnasep_data['avg_coverage'].median()
                    rnasep_ref_median = np.median([ref_control_medians.get(e, 1) for e in rnasep_data['exon']])
                    if rnasep_ref_median > 0:
                        rnasep_factor = rnasep_median / rnasep_ref_median
                
                normalization_factors[sample_id] = {
                    'global_factor': global_factor,
                    'cftr_factor': cftr_factor,
                    'rnasep_factor': rnasep_factor,
                    'combined_factor': np.median([global_factor, cftr_factor, rnasep_factor]),
                    'method': 'control_gene_based',
                    'control_genes_available': True,
                    'n_control_ratios': len(ratios)
                }
            else:
                # Fallback if no valid control ratios
                sample_target = target_coverage_df[target_coverage_df['sample_id'] == sample_id]
                median_coverage = sample_target['avg_coverage'].median() if not sample_target.empty else 1.0
                normalization_factors[sample_id] = {
                    'global_factor': median_coverage,
                    'method': 'target_fallback',
                    'control_genes_available': False
                }
        else:
            # Fallback to target gene normalization
            sample_target = target_coverage_df[target_coverage_df['sample_id'] == sample_id]
            median_coverage = sample_target['avg_coverage'].median() if not sample_target.empty else 1.0
            normalization_factors[sample_id] = {
                'global_factor': median_coverage,
                'method': 'target_fallback',
                'control_genes_available': False
            }
    
    return normalization_factors

def calculate_enhanced_reference_stats(coverage_df, reference_samples, normalization_factors):
    """Calculate enhanced reference statistics with control gene normalization."""
    ref_stats = {}
    
    # Filter for reference samples only
    ref_df = coverage_df[coverage_df['sample_id'].isin(reference_samples)]
    
    if ref_df.empty:
        print("Warning: No reference samples found in coverage data!")
        return ref_stats
    
    # Calculate statistics for each exon
    exons = ref_df['exon'].unique()
    
    for exon in exons:
        exon_data = ref_df[ref_df['exon'] == exon].copy()
        
        # Apply normalization factors
        normalized_coverage = []
        for _, row in exon_data.iterrows():
            sample_id = row['sample_id']
            raw_coverage = row['avg_coverage']
            
            if sample_id in normalization_factors:
                norm_factor = normalization_factors[sample_id]['combined_factor']
                if norm_factor > 0:
                    normalized_cov = raw_coverage / norm_factor
                else:
                    normalized_cov = raw_coverage
            else:
                normalized_cov = raw_coverage
            
            normalized_coverage.append(normalized_cov)
        
        normalized_coverage = np.array(normalized_coverage)
        
        if len(normalized_coverage) > 0:
            # Robust outlier removal using MAD (Median Absolute Deviation)
            median_cov = np.median(normalized_coverage)
            mad = np.median(np.abs(normalized_coverage - median_cov))
            
            # Modified Z-score threshold
            if mad > 0:
                modified_z_scores = 0.6745 * (normalized_coverage - median_cov) / mad
                outlier_mask = np.abs(modified_z_scores) < 3.5  # Less stringent for WES data
            else:
                outlier_mask = np.ones(len(normalized_coverage), dtype=bool)
            
            filtered_coverage = normalized_coverage[outlier_mask]
            
            if len(filtered_coverage) > 1:
                ref_stats[exon] = {
                    'mean': np.mean(filtered_coverage),
                    'std': np.std(filtered_coverage, ddof=1),
                    'median': np.median(filtered_coverage),
                    'mad': np.median(np.abs(filtered_coverage - np.median(filtered_coverage))),
                    'n_samples': len(filtered_coverage),
                    'n_outliers': len(normalized_coverage) - len(filtered_coverage),
                    'min': np.min(filtered_coverage),
                    'max': np.max(filtered_coverage),
                    'q25': np.percentile(filtered_coverage, 25),
                    'q75': np.percentile(filtered_coverage, 75),
                    'normalization_method': 'control_gene_enhanced'
                }
            else:
                # Fallback for small sample sizes
                ref_stats[exon] = {
                    'mean': median_cov,
                    'std': mad * 1.4826 if mad > 0 else 1.0,  # Convert MAD to approximate std
                    'median': median_cov,
                    'mad': mad,
                    'n_samples': len(normalized_coverage),
                    'n_outliers': 0,
                    'min': np.min(normalized_coverage),
                    'max': np.max(normalized_coverage),
                    'q25': median_cov,
                    'q75': median_cov,
                    'normalization_method': 'control_gene_enhanced_fallback'
                }
        else:
            print(f"Warning: No coverage data for exon {exon} in reference samples")
    
    return ref_stats

def calculate_enhanced_z_scores(coverage_df, ref_stats, normalization_factors):
    """Calculate enhanced Z-scores with batch effect correction."""
    z_score_data = []
    
    for _, row in coverage_df.iterrows():
        sample_id = row['sample_id']
        exon = row['exon']
        raw_coverage = row['avg_coverage']
        
        # Apply normalization
        if sample_id in normalization_factors:
            norm_factor = normalization_factors[sample_id]['combined_factor']
            if norm_factor > 0:
                normalized_coverage = raw_coverage / norm_factor
            else:
                normalized_coverage = raw_coverage
        else:
            normalized_coverage = raw_coverage
        
        if exon in ref_stats:
            ref_mean = ref_stats[exon]['mean']
            ref_std = ref_stats[exon]['std']
            
            # Calculate different types of Z-scores
            if ref_std > 0:
                z_score = (normalized_coverage - ref_mean) / ref_std
                
                # Robust Z-score using MAD
                ref_median = ref_stats[exon]['median']
                ref_mad = ref_stats[exon]['mad']
                if ref_mad > 0:
                    robust_z_score = 0.6745 * (normalized_coverage - ref_median) / ref_mad
                else:
                    robust_z_score = 0.0
            else:
                z_score = 0.0
                robust_z_score = 0.0
            
            # Quality metrics
            coverage_ratio = raw_coverage / ref_mean if ref_mean > 0 else 1.0
            normalization_quality = 'high' if sample_id in normalization_factors and normalization_factors[sample_id]['control_genes_available'] else 'medium'
            
            z_score_data.append({
                'sample_id': sample_id,
                'exon': exon,
                'raw_coverage': raw_coverage,
                'normalized_coverage': normalized_coverage,
                'normalization_factor': normalization_factors.get(sample_id, {}).get('combined_factor', 1.0),
                'ref_mean': ref_mean,
                'ref_std': ref_std,
                'ref_median': ref_stats[exon]['median'],
                'ref_mad': ref_stats[exon]['mad'],
                'z_score': z_score,
                'robust_z_score': robust_z_score,
                'coverage_ratio': coverage_ratio,
                'ref_n_samples': ref_stats[exon]['n_samples'],
                'normalization_quality': normalization_quality,
                'normalization_method': ref_stats[exon]['normalization_method']
            })
        else:
            print(f"Warning: No reference statistics for exon {exon}")
            z_score_data.append({
                'sample_id': sample_id,
                'exon': exon,
                'raw_coverage': raw_coverage,
                'normalized_coverage': normalized_coverage,
                'normalization_factor': 1.0,
                'ref_mean': np.nan,
                'ref_std': np.nan,
                'ref_median': np.nan,
                'ref_mad': np.nan,
                'z_score': np.nan,
                'robust_z_score': np.nan,
                'coverage_ratio': np.nan,
                'ref_n_samples': 0,
                'normalization_quality': 'low',
                'normalization_method': 'none'
            })
    
    return pd.DataFrame(z_score_data)

def create_enhanced_normalization_plots(z_scores_df, normalization_factors, ref_stats, output_dir):
    """Create enhanced visualization plots for the normalization results."""
    plot_dir = Path(output_dir) / 'plots'
    plot_dir.mkdir(exist_ok=True)
    
    # Plot 1: Normalization factors distribution
    plt.figure(figsize=(15, 10))
    
    # Global factors
    plt.subplot(2, 3, 1)
    factors = [nf.get('global_factor', 1.0) for nf in normalization_factors.values()]
    plt.hist(factors, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    plt.title('Global Normalization Factors')
    plt.xlabel('Normalization Factor')
    plt.ylabel('Frequency')
    
    # CFTR factors
    plt.subplot(2, 3, 2)
    cftr_factors = [nf.get('cftr_factor', 1.0) for nf in normalization_factors.values() if nf.get('control_genes_available', False)]
    if cftr_factors:
        plt.hist(cftr_factors, bins=20, alpha=0.7, edgecolor='black', color='orange')
        plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    plt.title('CFTR Normalization Factors')
    plt.xlabel('CFTR Factor')
    plt.ylabel('Frequency')
    
    # RNase P factors
    plt.subplot(2, 3, 3)
    rnasep_factors = [nf.get('rnasep_factor', 1.0) for nf in normalization_factors.values() if nf.get('control_genes_available', False)]
    if rnasep_factors:
        plt.hist(rnasep_factors, bins=20, alpha=0.7, edgecolor='black', color='green')
        plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    plt.title('RNase P Normalization Factors')
    plt.xlabel('RNase P Factor')
    plt.ylabel('Frequency')
    
    # Before vs After normalization comparison
    plt.subplot(2, 3, 4)
    exons = z_scores_df['exon'].unique()
    if len(exons) > 0:
        sample_exon = exons[0]  # Pick first exon for comparison
        exon_data = z_scores_df[z_scores_df['exon'] == sample_exon]
        plt.scatter(exon_data['raw_coverage'], exon_data['normalized_coverage'], alpha=0.6)
        plt.plot([0, exon_data['raw_coverage'].max()], [0, exon_data['raw_coverage'].max()], 'r--', alpha=0.7)
        plt.xlabel('Raw Coverage')
        plt.ylabel('Normalized Coverage')
        plt.title(f'Normalization Effect - {sample_exon}')
    
    # Z-score comparison
    plt.subplot(2, 3, 5)
    plt.scatter(z_scores_df['z_score'], z_scores_df['robust_z_score'], alpha=0.6)
    plt.plot([z_scores_df['z_score'].min(), z_scores_df['z_score'].max()], 
             [z_scores_df['z_score'].min(), z_scores_df['z_score'].max()], 'r--', alpha=0.7)
    plt.xlabel('Standard Z-score')
    plt.ylabel('Robust Z-score')
    plt.title('Z-score Method Comparison')
    
    # Normalization quality distribution
    plt.subplot(2, 3, 6)
    quality_counts = z_scores_df['normalization_quality'].value_counts()
    plt.pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Normalization Quality Distribution')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'enhanced_normalization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Enhanced Z-score heatmaps
    plt.figure(figsize=(15, 8))
    
    # Standard Z-scores
    plt.subplot(1, 2, 1)
    pivot_z = z_scores_df.pivot(index='sample_id', columns='exon', values='z_score')
    sns.heatmap(pivot_z, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Standard Z-score'}, 
                fmt='.2f', linewidths=0.5)
    plt.title('Standard Z-scores Heatmap')
    plt.ylabel('Sample ID')
    plt.xlabel('Exon')
    
    # Robust Z-scores
    plt.subplot(1, 2, 2)
    pivot_robust = z_scores_df.pivot(index='sample_id', columns='exon', values='robust_z_score')
    sns.heatmap(pivot_robust, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Robust Z-score'}, 
                fmt='.2f', linewidths=0.5)
    plt.title('Robust Z-scores Heatmap')
    plt.ylabel('Sample ID')
    plt.xlabel('Exon')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'enhanced_z_score_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced normalization plots saved to: {plot_dir}")

def main():
    if len(sys.argv) != 5:
        print("Usage: python control_gene_normalization.py <coverage_file> <sample_info_file> <control_coverage_file> <output_file>")
        print("  coverage_file: Target gene coverage data file")
        print("  sample_info_file: Sample information file")
        print("  control_coverage_file: Control gene coverage file (CFTR and RNase P)")
        print("  output_file: Output file for enhanced normalized data")
        sys.exit(1)
    
    coverage_file = sys.argv[1]
    sample_info_file = sys.argv[2]
    control_coverage_file = sys.argv[3]
    output_file = sys.argv[4]
    
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Reading input files...")
    
    # Read coverage data
    try:
        coverage_df = pd.read_csv(coverage_file, sep='\t')
        print(f"Loaded target coverage data for {len(coverage_df)} sample-exon combinations")
    except Exception as e:
        print(f"Error reading coverage file: {e}")
        sys.exit(1)
    
    # Read sample information
    samples_info = read_sample_info(sample_info_file)
    
    # Read control gene coverage
    control_coverage_df = load_control_gene_coverage(control_coverage_file)
    
    # Auto-detect sample types if needed
    if not samples_info:
        print("Auto-detecting sample types from sample names...")
        for sample_id in coverage_df['sample_id'].unique():
            if any(keyword in sample_id.lower() for keyword in ['ref', 'control', 'normal']):
                sample_type = 'reference'
            else:
                sample_type = 'test'
            samples_info[sample_id] = {'sample_type': sample_type, 'bam_path': ''}
    
    # Identify reference samples
    reference_samples = [sid for sid, info in samples_info.items() 
                        if info['sample_type'] == 'reference']
    
    print(f"Found {len(reference_samples)} reference samples")
    
    if len(reference_samples) < 2:
        print("Warning: Very few reference samples. Results may be unreliable.")
    
    # Calculate control gene-based normalization factors
    print("Calculating control gene-based normalization factors...")
    normalization_factors = calculate_control_gene_factors(coverage_df, control_coverage_df, reference_samples)
    
    # Calculate enhanced reference statistics
    print("Calculating enhanced reference statistics...")
    ref_stats = calculate_enhanced_reference_stats(coverage_df, reference_samples, normalization_factors)
    
    # Calculate enhanced Z-scores
    print("Calculating enhanced Z-scores...")
    z_scores_df = calculate_enhanced_z_scores(coverage_df, ref_stats, normalization_factors)
    
    # Add sample type information
    z_scores_df['sample_type'] = z_scores_df['sample_id'].map(
        lambda x: samples_info.get(x, {}).get('sample_type', 'unknown')
    )
    z_scores_df['population'] = 'unknown'  # Placeholder for population info
    
    # Save results
    z_scores_df.to_csv(output_file, index=False, sep='\t')
    
    # Save enhanced reference statistics
    ref_stats_file = output_file.replace('.txt', '_enhanced_ref_stats.txt')
    ref_stats_df = pd.DataFrame.from_dict(ref_stats, orient='index').reset_index()
    ref_stats_df.rename(columns={'index': 'exon'}, inplace=True)
    ref_stats_df.to_csv(ref_stats_file, index=False, sep='\t')
    
    # Save normalization factors
    norm_factors_file = output_file.replace('.txt', '_normalization_factors.txt')
    norm_df_data = []
    for sample_id, factors in normalization_factors.items():
        norm_df_data.append({
            'sample_id': sample_id,
            'global_factor': factors.get('global_factor', 1.0),
            'cftr_factor': factors.get('cftr_factor', 1.0),
            'rnasep_factor': factors.get('rnasep_factor', 1.0),
            'combined_factor': factors.get('combined_factor', 1.0),
            'method': factors.get('method', 'unknown'),
            'control_genes_available': factors.get('control_genes_available', False),
            'n_control_ratios': factors.get('n_control_ratios', 0)
        })
    norm_df = pd.DataFrame(norm_df_data)
    norm_df.to_csv(norm_factors_file, index=False, sep='\t')
    
    # Create enhanced plots
    try:
        create_enhanced_normalization_plots(z_scores_df, normalization_factors, ref_stats, output_dir)
    except Exception as e:
        print(f"Warning: Could not create enhanced plots: {e}")
    
    # Print summary
    print(f"\nEnhanced normalization completed!")
    print(f"Z-scores saved to: {output_file}")
    print(f"Enhanced reference statistics saved to: {ref_stats_file}")
    print(f"Normalization factors saved to: {norm_factors_file}")
    
    print(f"\nSummary statistics:")
    print(f"Total samples: {len(z_scores_df['sample_id'].unique())}")
    print(f"Reference samples: {len(reference_samples)}")
    print(f"Control gene normalization available: {sum(1 for nf in normalization_factors.values() if nf.get('control_genes_available', False))}")
    
    print(f"\nEnhanced Z-score summary by exon:")
    summary = z_scores_df.groupby('exon')[['z_score', 'robust_z_score']].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
    print(summary)
    
    print(f"\nNormalization quality distribution:")
    quality_dist = z_scores_df['normalization_quality'].value_counts()
    print(quality_dist)

if __name__ == "__main__":
    main()
