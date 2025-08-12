#!/usr/bin/env python3

"""
estimate_copy_number.py - Estimate copy number states from Z-scores using predefined thresholds
Usage: python estimate_copy_number.py <z_scores_file> <output_file> [--thresholds custom_thresholds.txt]
"""

import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Default copy number thresholds based on Z-scores
DEFAULT_THRESHOLDS = {
    'homozygous_deletion': -2.5,    # CN=0
    'heterozygous_deletion': -1.5,  # CN=1
    'normal_lower': -1.5,           # CN=2 lower bound
    'normal_upper': 1.5,            # CN=2 upper bound
    'duplication': 2.5,             # CN=3
    # CN=4+ is anything above duplication threshold
}

def read_custom_thresholds(threshold_file):
    """Read custom thresholds from file."""
    thresholds = DEFAULT_THRESHOLDS.copy()
    
    try:
        with open(threshold_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    threshold_name, value = parts[0], float(parts[1])
                    if threshold_name in thresholds:
                        thresholds[threshold_name] = value
        
        print(f"Loaded custom thresholds from: {threshold_file}")
    except Exception as e:
        print(f"Warning: Could not read custom thresholds ({e}). Using defaults.")
    
    return thresholds

def assign_copy_number(z_score, thresholds):
    """Assign copy number based on Z-score and thresholds."""
    if pd.isna(z_score):
        return {
            'copy_number': np.nan,
            'cn_category': 'unknown',
            'confidence': 'low'
        }
    
    # Define copy number based on thresholds
    if z_score <= thresholds['homozygous_deletion']:
        cn = 0
        category = 'homozygous_deletion'
    elif z_score <= thresholds['heterozygous_deletion']:
        cn = 1
        category = 'heterozygous_deletion'
    elif thresholds['normal_lower'] < z_score <= thresholds['normal_upper']:
        cn = 2
        category = 'normal'
    elif z_score <= thresholds['duplication']:
        cn = 3
        category = 'duplication'
    else:
        cn = 4  # Could be higher, but we cap at 4+
        category = 'high_amplification'
    
    # Assign confidence based on how far from boundaries
    confidence = 'high'
    boundary_distances = [
        abs(z_score - thresholds['homozygous_deletion']),
        abs(z_score - thresholds['heterozygous_deletion']),
        abs(z_score - thresholds['normal_lower']),
        abs(z_score - thresholds['normal_upper']),
        abs(z_score - thresholds['duplication'])
    ]
    
    min_distance = min(boundary_distances)
    if min_distance < 0.5:
        confidence = 'low'
    elif min_distance < 1.0:
        confidence = 'medium'
    
    return {
        'copy_number': cn,
        'cn_category': category,
        'confidence': confidence
    }

def estimate_gene_copy_numbers(cn_results_df):
    """Estimate overall gene copy numbers from exon-level results."""
    gene_results = []
    
    # Group by sample and gene (inferred from exon names)
    for sample_id in cn_results_df['sample_id'].unique():
        sample_data = cn_results_df[cn_results_df['sample_id'] == sample_id]
        
        # Separate SMN1 and SMN2 based on exon names
        smn1_data = sample_data[sample_data['exon'].str.contains('SMN1')]
        smn2_data = sample_data[sample_data['exon'].str.contains('SMN2')]
        
        for gene_name, gene_data in [('SMN1', smn1_data), ('SMN2', smn2_data)]:
            if not gene_data.empty:
                # Use median copy number across exons for gene-level estimate
                median_cn = gene_data['copy_number'].median()
                mean_z_score = gene_data['z_score'].mean()
                
                # Confidence based on consistency across exons
                cn_values = gene_data['copy_number'].dropna()
                if len(cn_values) > 1:
                    cn_std = cn_values.std()
                    if cn_std == 0:
                        gene_confidence = 'high'
                    elif cn_std <= 0.5:
                        gene_confidence = 'medium'
                    else:
                        gene_confidence = 'low'
                else:
                    gene_confidence = 'medium'
                
                # Determine category from median CN
                if median_cn == 0:
                    gene_category = 'homozygous_deletion'
                elif median_cn == 1:
                    gene_category = 'heterozygous_deletion'
                elif median_cn == 2:
                    gene_category = 'normal'
                elif median_cn == 3:
                    gene_category = 'duplication'
                else:
                    gene_category = 'high_amplification'
                
                gene_results.append({
                    'sample_id': sample_id,
                    'gene': gene_name,
                    'estimated_copy_number': median_cn,
                    'mean_z_score': mean_z_score,
                    'cn_category': gene_category,
                    'confidence': gene_confidence,
                    'n_exons': len(gene_data),
                    'exon_cn_std': cn_values.std() if len(cn_values) > 1 else 0
                })
    
    return pd.DataFrame(gene_results)

def create_cn_visualization(cn_results_df, gene_results_df, output_dir, thresholds):
    """Create visualization plots for copy number results."""
    plot_dir = Path(output_dir) / 'plots'
    plot_dir.mkdir(exist_ok=True)
    
    # Plot 1: Copy number distribution by exon
    plt.figure(figsize=(12, 8))
    exons = sorted(cn_results_df['exon'].unique())
    
    for i, exon in enumerate(exons):
        plt.subplot(2, 2, i+1)
        exon_data = cn_results_df[cn_results_df['exon'] == exon]
        
        cn_counts = exon_data['copy_number'].value_counts().sort_index()
        plt.bar(cn_counts.index, cn_counts.values, alpha=0.7)
        plt.title(f'{exon} Copy Number Distribution')
        plt.xlabel('Copy Number')
        plt.ylabel('Sample Count')
        plt.xticks(range(0, 5))
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'copy_number_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Z-score vs Copy Number with thresholds
    plt.figure(figsize=(15, 10))
    
    for i, exon in enumerate(exons):
        plt.subplot(2, 2, i+1)
        exon_data = cn_results_df[cn_results_df['exon'] == exon]
        
        # Scatter plot colored by copy number
        scatter = plt.scatter(exon_data.index, exon_data['z_score'], 
                            c=exon_data['copy_number'], cmap='viridis',
                            alpha=0.7, s=50)
        
        # Add threshold lines
        plt.axhline(y=thresholds['homozygous_deletion'], color='red', 
                   linestyle='--', alpha=0.7, label='CN=0/1')
        plt.axhline(y=thresholds['heterozygous_deletion'], color='orange', 
                   linestyle='--', alpha=0.7, label='CN=1/2')
        plt.axhline(y=thresholds['normal_upper'], color='green', 
                   linestyle='--', alpha=0.7, label='CN=2/3')
        plt.axhline(y=thresholds['duplication'], color='purple', 
                   linestyle='--', alpha=0.7, label='CN=3/4+')
        
        plt.title(f'{exon}: Z-scores with CN Thresholds')
        plt.xlabel('Sample Index')
        plt.ylabel('Z-score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.colorbar(scatter, label='Copy Number')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'zscore_vs_copy_number.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Gene-level copy number heatmap
    if not gene_results_df.empty:
        gene_pivot = gene_results_df.pivot(index='sample_id', columns='gene', 
                                         values='estimated_copy_number')
        
        plt.figure(figsize=(8, max(6, len(gene_pivot) * 0.3)))
        sns.heatmap(gene_pivot, annot=True, cmap='RdYlBu_r', center=2,
                   cbar_kws={'label': 'Copy Number'}, 
                   fmt='.1f', linewidths=0.5)
        plt.title('Gene-level Copy Number Estimates')
        plt.ylabel('Sample ID')
        plt.xlabel('Gene')
        plt.tight_layout()
        plt.savefig(plot_dir / 'gene_copy_number_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Copy number plots saved to: {plot_dir}")

def main():
    parser = argparse.ArgumentParser(description='Estimate copy numbers from Z-scores')
    parser.add_argument('z_scores_file', help='Z-scores file from normalize_coverage.py')
    parser.add_argument('output_file', help='Output file for copy number estimates')
    parser.add_argument('--thresholds', help='Custom thresholds file', default=None)
    parser.add_argument('--no-plots', action='store_true', help='Skip creating plots')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load thresholds
    if args.thresholds:
        thresholds = read_custom_thresholds(args.thresholds)
    else:
        thresholds = DEFAULT_THRESHOLDS.copy()
        print("Using default thresholds")
    
    print(f"Copy number thresholds:")
    for name, value in thresholds.items():
        print(f"  {name}: {value}")
    
    # Read Z-scores
    try:
        z_scores_df = pd.read_csv(args.z_scores_file, sep='\t')
        print(f"Loaded Z-scores for {len(z_scores_df)} sample-exon combinations")
    except Exception as e:
        print(f"Error reading Z-scores file: {e}")
        sys.exit(1)
    
    # Estimate copy numbers
    print("Estimating copy numbers...")
    cn_results = []
    
    for _, row in z_scores_df.iterrows():
        cn_result = assign_copy_number(row['z_score'], thresholds)
        
        result = {
            'sample_id': row['sample_id'],
            'exon': row['exon'],
            'z_score': row['z_score'],
            'copy_number': cn_result['copy_number'],
            'cn_category': cn_result['cn_category'],
            'confidence': cn_result['confidence'],
            'raw_coverage': row.get('raw_coverage', np.nan),
            'sample_type': row.get('sample_type', 'unknown'),
            'population': row.get('population', 'unknown')
        }
        cn_results.append(result)
    
    cn_results_df = pd.DataFrame(cn_results)
    
    # Estimate gene-level copy numbers
    print("Estimating gene-level copy numbers...")
    gene_results_df = estimate_gene_copy_numbers(cn_results_df)
    
    # Save results
    cn_results_df.to_csv(args.output_file, index=False, sep='\t')
    
    gene_output_file = args.output_file.replace('.txt', '_gene_level.txt')
    gene_results_df.to_csv(gene_output_file, index=False, sep='\t')
    
    # Save thresholds used
    threshold_output_file = args.output_file.replace('.txt', '_thresholds.txt')
    with open(threshold_output_file, 'w') as f:
        f.write("# Copy number thresholds used\n")
        f.write("threshold_name\tvalue\tdescription\n")
        f.write(f"homozygous_deletion\t{thresholds['homozygous_deletion']}\tCN=0\n")
        f.write(f"heterozygous_deletion\t{thresholds['heterozygous_deletion']}\tCN=1\n")
        f.write(f"normal_lower\t{thresholds['normal_lower']}\tCN=2 lower bound\n")
        f.write(f"normal_upper\t{thresholds['normal_upper']}\tCN=2 upper bound\n")
        f.write(f"duplication\t{thresholds['duplication']}\tCN=3\n")
    
    # Create plots
    if not args.no_plots:
        try:
            create_cn_visualization(cn_results_df, gene_results_df, output_dir, thresholds)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
    
    # Print summary
    print(f"\nCopy number estimation completed!")
    print(f"Exon-level results saved to: {args.output_file}")
    print(f"Gene-level results saved to: {gene_output_file}")
    print(f"Thresholds saved to: {threshold_output_file}")
    
    print(f"\nExon-level copy number summary:")
    cn_summary = cn_results_df.groupby(['exon', 'copy_number']).size().unstack(fill_value=0)
    print(cn_summary)
    
    print(f"\nGene-level copy number summary:")
    if not gene_results_df.empty:
        gene_summary = gene_results_df.groupby(['gene', 'estimated_copy_number']).size().unstack(fill_value=0)
        print(gene_summary)
    
    print(f"\nConfidence distribution:")
    confidence_summary = cn_results_df['confidence'].value_counts()
    print(confidence_summary)

if __name__ == "__main__":
    main()
