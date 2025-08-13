#!/usr/bin/env python3

"""
enhanced_copy_number_estimation.py - Enhanced copy number estimation with dual segmentation integration
Usage: python enhanced_copy_number_estimation.py <segmentation_file> <output_file> [--thresholds custom_thresholds.txt]
"""

import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Enhanced copy number thresholds with confidence intervals
DEFAULT_THRESHOLDS = {
    'homozygous_deletion': -2.5,    # CN=0
    'heterozygous_deletion': -1.5,  # CN=1
    'normal_lower': -1.5,           # CN=2 lower bound
    'normal_upper': 1.5,            # CN=2 upper bound
    'duplication': 2.5,             # CN=3
    # Confidence thresholds
    'high_confidence': 0.5,         # Distance from boundary for high confidence
    'medium_confidence': 1.0,       # Distance from boundary for medium confidence
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    'min_coverage': 10,             # Minimum coverage for reliable calls
    'max_missing_rate': 0.3,        # Maximum missing data rate per sample
    'min_segment_size': 2,          # Minimum segment size for reliable calls
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

def calculate_enhanced_confidence(z_score, copy_number, thresholds, quality_metrics):
    """Calculate enhanced confidence score for copy number call."""
    if pd.isna(z_score) or pd.isna(copy_number):
        return 'unknown', 0.0
    
    # Calculate distance to nearest boundary
    boundaries = [
        thresholds['homozygous_deletion'],
        thresholds['heterozygous_deletion'],
        thresholds['normal_lower'],
        thresholds['normal_upper'],
        thresholds['duplication']
    ]
    
    boundary_distances = [abs(z_score - b) for b in boundaries]
    min_distance = min(boundary_distances)
    
    # Base confidence from Z-score distance
    if min_distance >= thresholds['medium_confidence']:
        base_confidence = 'high'
        confidence_score = 0.9
    elif min_distance >= thresholds['high_confidence']:
        base_confidence = 'medium'
        confidence_score = 0.7
    else:
        base_confidence = 'low'
        confidence_score = 0.4
    
    # Adjust confidence based on quality metrics
    coverage_penalty = 0.0
    if quality_metrics.get('raw_coverage', 0) < QUALITY_THRESHOLDS['min_coverage']:
        coverage_penalty += 0.2
    
    normalization_penalty = 0.0
    if quality_metrics.get('normalization_quality', 'medium') == 'low':
        normalization_penalty += 0.1
    
    missing_data_penalty = 0.0
    if quality_metrics.get('missing_rate', 0) > QUALITY_THRESHOLDS['max_missing_rate']:
        missing_data_penalty += 0.15
    
    # Segmentation consistency bonus
    segmentation_bonus = 0.0
    if quality_metrics.get('segmentation_method', '') == 'consensus':
        segmentation_bonus += 0.1
    elif quality_metrics.get('segmentation_method', '') in ['hmm', 'cbs']:
        segmentation_bonus += 0.05
    
    # Final confidence score
    final_confidence_score = max(0.0, min(1.0, 
        confidence_score - coverage_penalty - normalization_penalty - missing_data_penalty + segmentation_bonus
    ))
    
    # Final confidence category
    if final_confidence_score >= 0.8:
        final_confidence = 'high'
    elif final_confidence_score >= 0.6:
        final_confidence = 'medium'
    else:
        final_confidence = 'low'
    
    return final_confidence, final_confidence_score

def assign_enhanced_copy_number(z_score, robust_z_score, thresholds, quality_metrics=None):
    """Assign copy number with enhanced logic using both standard and robust Z-scores."""
    if quality_metrics is None:
        quality_metrics = {}
    
    if pd.isna(z_score) and pd.isna(robust_z_score):
        return {
            'copy_number': np.nan,
            'cn_category': 'unknown',
            'confidence': 'low',
            'confidence_score': 0.0,
            'primary_score': np.nan,
            'method': 'missing_data'
        }
    
    # Use robust Z-score preferentially, fall back to standard Z-score
    if not pd.isna(robust_z_score):
        primary_score = robust_z_score
        method = 'robust_z_score'
    else:
        primary_score = z_score
        method = 'standard_z_score'
    
    # Enhanced copy number assignment with fuzzy boundaries
    if primary_score <= thresholds['homozygous_deletion']:
        cn = 0
        category = 'homozygous_deletion'
    elif primary_score <= thresholds['heterozygous_deletion']:
        cn = 1
        category = 'heterozygous_deletion'
    elif thresholds['normal_lower'] < primary_score <= thresholds['normal_upper']:
        cn = 2
        category = 'normal'
    elif primary_score <= thresholds['duplication']:
        cn = 3
        category = 'duplication'
    else:
        cn = 4  # 4+ amplifications
        category = 'high_amplification'
    
    # Handle boundary cases with additional validation
    if not pd.isna(z_score) and not pd.isna(robust_z_score):
        # Check agreement between methods
        z_cn = assign_cn_from_score(z_score, thresholds)
        robust_cn = assign_cn_from_score(robust_z_score, thresholds)
        
        if z_cn != robust_cn:
            # Methods disagree - use more conservative call
            if abs(z_cn - 2) < abs(robust_cn - 2):
                cn = z_cn
                method = 'conservative_standard'
            else:
                cn = robust_cn
                method = 'conservative_robust'
            category = cn_to_category(cn)
    
    # Calculate confidence
    confidence, confidence_score = calculate_enhanced_confidence(
        primary_score, cn, thresholds, quality_metrics
    )
    
    return {
        'copy_number': cn,
        'cn_category': category,
        'confidence': confidence,
        'confidence_score': confidence_score,
        'primary_score': primary_score,
        'method': method
    }

def assign_cn_from_score(score, thresholds):
    """Helper function to assign CN from a single score."""
    if score <= thresholds['homozygous_deletion']:
        return 0
    elif score <= thresholds['heterozygous_deletion']:
        return 1
    elif thresholds['normal_lower'] < score <= thresholds['normal_upper']:
        return 2
    elif score <= thresholds['duplication']:
        return 3
    else:
        return 4

def cn_to_category(cn):
    """Convert copy number to category."""
    categories = {
        0: 'homozygous_deletion',
        1: 'heterozygous_deletion',
        2: 'normal',
        3: 'duplication',
        4: 'high_amplification'
    }
    return categories.get(cn, 'unknown')

def estimate_enhanced_gene_copy_numbers(cn_results_df):
    """Estimate gene-level copy numbers with enhanced quality metrics."""
    gene_results = []
    
    # Group by sample and gene
    for sample_id in cn_results_df['sample_id'].unique():
        sample_data = cn_results_df[cn_results_df['sample_id'] == sample_id]
        
        # Separate SMN1 and SMN2 based on exon names
        smn1_data = sample_data[sample_data['exon'].str.contains('SMN1')]
        smn2_data = sample_data[sample_data['exon'].str.contains('SMN2')]
        
        for gene_name, gene_data in [('SMN1', smn1_data), ('SMN2', smn2_data)]:
            if not gene_data.empty:
                # Enhanced gene-level estimation
                cn_values = gene_data['copy_number'].dropna()
                confidence_scores = gene_data['confidence_score'].dropna()
                
                if len(cn_values) > 0:
                    # Weighted median based on confidence scores
                    if len(confidence_scores) == len(cn_values) and confidence_scores.sum() > 0:
                        # Weight by confidence
                        weights = confidence_scores / confidence_scores.sum()
                        estimated_cn = np.average(cn_values, weights=weights)
                        estimated_cn = round(estimated_cn)  # Round to nearest integer
                    else:
                        estimated_cn = np.median(cn_values)
                    
                    # Gene-level confidence
                    mean_confidence_score = np.mean(confidence_scores) if len(confidence_scores) > 0 else 0.5
                    cn_consistency = len(set(cn_values)) == 1  # All exons agree
                    
                    # Quality metrics
                    coverage_metrics = {
                        'mean_coverage': gene_data['raw_coverage'].mean(),
                        'min_coverage': gene_data['raw_coverage'].min(),
                        'coverage_cv': gene_data['raw_coverage'].std() / gene_data['raw_coverage'].mean() if gene_data['raw_coverage'].mean() > 0 else np.nan
                    }
                    
                    # Missing data rate
                    missing_rate = gene_data[['z_score', 'robust_z_score']].isna().all(axis=1).mean()
                    
                    # Segmentation quality
                    segmentation_methods = gene_data['segmentation_method'].unique()
                    segmentation_quality = 'high' if 'consensus' in segmentation_methods else 'medium' if len(segmentation_methods) == 1 else 'low'
                    
                    # Final gene confidence
                    if cn_consistency and mean_confidence_score >= 0.8 and missing_rate < 0.2:
                        gene_confidence = 'high'
                    elif mean_confidence_score >= 0.6 and missing_rate < 0.4:
                        gene_confidence = 'medium'
                    else:
                        gene_confidence = 'low'
                    
                    # Clinical significance
                    clinical_significance = determine_clinical_significance(gene_name, estimated_cn, gene_confidence)
                    
                    gene_results.append({
                        'sample_id': sample_id,
                        'gene': gene_name,
                        'estimated_copy_number': estimated_cn,
                        'mean_z_score': gene_data['z_score'].mean(),
                        'mean_robust_z_score': gene_data['robust_z_score'].mean(),
                        'cn_category': cn_to_category(int(estimated_cn)),
                        'confidence': gene_confidence,
                        'confidence_score': mean_confidence_score,
                        'n_exons': len(gene_data),
                        'exon_cn_consistency': cn_consistency,
                        'exon_cn_std': cn_values.std() if len(cn_values) > 1 else 0,
                        'mean_coverage': coverage_metrics['mean_coverage'],
                        'min_coverage': coverage_metrics['min_coverage'],
                        'coverage_cv': coverage_metrics['coverage_cv'],
                        'missing_data_rate': missing_rate,
                        'segmentation_quality': segmentation_quality,
                        'clinical_significance': clinical_significance
                    })
    
    return pd.DataFrame(gene_results)

def determine_clinical_significance(gene_name, copy_number, confidence):
    """Determine clinical significance of copy number call."""
    if gene_name == 'SMN1':
        if copy_number == 0:
            return 'likely_pathogenic' if confidence in ['high', 'medium'] else 'uncertain_significance'
        elif copy_number == 1:
            return 'likely_benign_carrier' if confidence in ['high', 'medium'] else 'uncertain_significance'
        elif copy_number == 2:
            return 'benign'
        elif copy_number >= 3:
            return 'likely_benign_duplication'
    elif gene_name == 'SMN2':
        if copy_number == 0:
            return 'uncertain_significance'  # SMN2 deletion alone is not pathogenic
        elif copy_number >= 3:
            return 'likely_benign_modifier'  # May modify SMA severity
        else:
            return 'benign'
    
    return 'uncertain_significance'

def create_enhanced_cnv_visualization(cn_results_df, gene_results_df, output_dir, thresholds):
    """Create enhanced visualization plots for copy number results."""
    plot_dir = Path(output_dir) / 'plots'
    plot_dir.mkdir(exist_ok=True)
    
    # Plot 1: Enhanced copy number distribution with confidence
    plt.figure(figsize=(15, 12))
    
    # Distribution by exon with confidence coloring
    plt.subplot(2, 3, 1)
    exons = sorted(cn_results_df['exon'].unique())
    confidence_colors = {'high': 'green', 'medium': 'orange', 'low': 'red', 'unknown': 'gray'}
    
    for i, exon in enumerate(exons):
        exon_data = cn_results_df[cn_results_df['exon'] == exon]
        bottom = 0
        
        for conf in ['high', 'medium', 'low', 'unknown']:
            conf_data = exon_data[exon_data['confidence'] == conf]
            cn_counts = conf_data['copy_number'].value_counts().sort_index()
            
            for cn, count in cn_counts.items():
                plt.bar(f'{exon}\nCN{cn}', count, bottom=bottom, 
                       color=confidence_colors[conf], alpha=0.7,
                       label=f'{conf.title()} Confidence' if i == 0 and cn == cn_counts.index[0] else '')
                bottom += count
    
    plt.title('Copy Number Distribution by Confidence')
    plt.xlabel('Exon and Copy Number')
    plt.ylabel('Sample Count')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Confidence score distribution
    plt.subplot(2, 3, 2)
    plt.hist(cn_results_df['confidence_score'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='High Confidence')
    plt.axvline(x=0.6, color='orange', linestyle='--', alpha=0.7, label='Medium Confidence')
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Z-score vs Copy Number with confidence
    plt.subplot(2, 3, 3)
    scatter = plt.scatter(cn_results_df['primary_score'], cn_results_df['copy_number'], 
                         c=cn_results_df['confidence_score'], cmap='RdYlGn',
                         alpha=0.7, s=50)
    
    # Add threshold lines
    for name, value in thresholds.items():
        if 'confidence' not in name:
            plt.axvline(x=value, linestyle='--', alpha=0.5, 
                       label=f'{name.replace("_", " ").title()}')
    
    plt.title('Z-scores vs Copy Number (colored by confidence)')
    plt.xlabel('Primary Z-score')
    plt.ylabel('Copy Number')
    plt.colorbar(scatter, label='Confidence Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Gene-level results with clinical significance
    plt.subplot(2, 3, 4)
    if not gene_results_df.empty:
        significance_colors = {
            'likely_pathogenic': 'red',
            'likely_benign_carrier': 'orange', 
            'benign': 'green',
            'likely_benign_duplication': 'blue',
            'likely_benign_modifier': 'purple',
            'uncertain_significance': 'gray'
        }
        
        for gene in ['SMN1', 'SMN2']:
            gene_data = gene_results_df[gene_results_df['gene'] == gene]
            if not gene_data.empty:
                x_offset = 0 if gene == 'SMN1' else 0.4
                
                for _, row in gene_data.iterrows():
                    color = significance_colors.get(row['clinical_significance'], 'gray')
                    plt.scatter(row['estimated_copy_number'] + x_offset, 
                              len(gene_data) - list(gene_data.index).index(row.name),
                              c=color, s=100, alpha=0.7,
                              label=row['clinical_significance'] if row['clinical_significance'] not in plt.gca().get_legend_handles_labels()[1] else '')
        
        plt.title('Gene-level CN with Clinical Significance')
        plt.xlabel('Copy Number')
        plt.ylabel('Sample Index')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Segmentation method performance
    plt.subplot(2, 3, 5)
    method_conf = cn_results_df.groupby(['segmentation_method', 'confidence']).size().unstack(fill_value=0)
    method_conf.plot(kind='bar', stacked=True, ax=plt.gca(), 
                    color=[confidence_colors[c] for c in method_conf.columns])
    plt.title('Segmentation Method Performance')
    plt.xlabel('Segmentation Method')
    plt.ylabel('Number of Calls')
    plt.xticks(rotation=45)
    plt.legend(title='Confidence', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Coverage vs Confidence
    plt.subplot(2, 3, 6)
    plt.scatter(cn_results_df['raw_coverage'], cn_results_df['confidence_score'], alpha=0.6)
    plt.axvline(x=QUALITY_THRESHOLDS['min_coverage'], color='red', 
               linestyle='--', alpha=0.7, label='Min Coverage Threshold')
    plt.title('Coverage vs Confidence Score')
    plt.xlabel('Raw Coverage')
    plt.ylabel('Confidence Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'enhanced_copy_number_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gene-level heatmap with confidence
    if not gene_results_df.empty:
        plt.figure(figsize=(12, max(8, len(gene_results_df) * 0.4)))
        
        # Prepare data for heatmap
        heatmap_data = gene_results_df.pivot_table(
            index='sample_id', 
            columns='gene', 
            values='estimated_copy_number',
            fill_value=np.nan
        )
        
        # Create confidence annotation
        confidence_data = gene_results_df.pivot_table(
            index='sample_id',
            columns='gene', 
            values='confidence_score',
            fill_value=0
        )
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', center=2,
                   cbar_kws={'label': 'Copy Number'}, 
                   fmt='.0f', linewidths=0.5)
        
        # Add confidence indicators as text overlay
        for i, row in enumerate(heatmap_data.index):
            for j, col in enumerate(heatmap_data.columns):
                if not pd.isna(heatmap_data.loc[row, col]):
                    conf_score = confidence_data.loc[row, col]
                    if conf_score >= 0.8:
                        marker = '●'  # High confidence
                    elif conf_score >= 0.6:
                        marker = '◐'  # Medium confidence  
                    else:
                        marker = '○'  # Low confidence
                    
                    plt.text(j + 0.8, i + 0.2, marker, fontsize=12, 
                            color='white' if heatmap_data.loc[row, col] <= 2 else 'black')
        
        plt.title('Gene-level Copy Numbers with Confidence Indicators\n(● High, ◐ Medium, ○ Low)')
        plt.ylabel('Sample ID')
        plt.xlabel('Gene')
        plt.tight_layout()
        plt.savefig(plot_dir / 'gene_level_cnv_heatmap_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Enhanced CNV analysis plots saved to: {plot_dir}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced copy number estimation with dual segmentation')
    parser.add_argument('segmentation_file', help='Dual segmentation results file')
    parser.add_argument('output_file', help='Output file for enhanced copy number estimates')
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
        print("Using default enhanced thresholds")
    
    print(f"Enhanced copy number thresholds:")
    for name, value in thresholds.items():
        print(f"  {name}: {value}")
    
    # Read segmentation results
    try:
        segmentation_df = pd.read_csv(args.segmentation_file, sep='\t')
        print(f"Loaded segmentation results for {len(segmentation_df)} sample-exon combinations")
    except Exception as e:
        print(f"Error reading segmentation file: {e}")
        sys.exit(1)
    
    # Enhanced copy number estimation
    print("Performing enhanced copy number estimation...")
    enhanced_cn_results = []
    
    for _, row in segmentation_df.iterrows():
        # Prepare quality metrics
        quality_metrics = {
            'raw_coverage': row.get('raw_coverage', 0),
            'normalized_coverage': row.get('normalized_coverage', 0),
            'normalization_quality': row.get('normalization_quality', 'medium'),
            'segmentation_method': row.get('segmentation_method', 'unknown'),
            'missing_rate': 0.0  # Calculate if needed
        }
        
        # Calculate missing data rate for this sample-exon
        if pd.isna(row.get('z_score')) and pd.isna(row.get('robust_z_score')):
            quality_metrics['missing_rate'] = 1.0
        elif pd.isna(row.get('z_score')) or pd.isna(row.get('robust_z_score')):
            quality_metrics['missing_rate'] = 0.5
        
        # Enhanced copy number assignment
        cn_result = assign_enhanced_copy_number(
            row.get('z_score'), 
            row.get('robust_z_score'), 
            thresholds, 
            quality_metrics
        )
        
        # Prepare enhanced result record
        enhanced_result = {
            'sample_id': row['sample_id'],
            'exon': row['exon'],
            'z_score': row.get('z_score', np.nan),
            'robust_z_score': row.get('robust_z_score', np.nan),
            'primary_score': cn_result['primary_score'],
            'copy_number': cn_result['copy_number'],
            'cn_category': cn_result['cn_category'],
            'confidence': cn_result['confidence'],
            'confidence_score': cn_result['confidence_score'],
            'assignment_method': cn_result['method'],
            'raw_coverage': row.get('raw_coverage', np.nan),
            'normalized_coverage': row.get('normalized_coverage', np.nan),
            'normalization_factor': row.get('normalization_factor', 1.0),
            'normalization_quality': quality_metrics['normalization_quality'],
            'segmentation_method': quality_metrics['segmentation_method'],
            'segmentation_available': row.get('cbs_available', False) or row.get('hmm_available', False),
            'sample_type': row.get('sample_type', 'unknown'),
            'population': row.get('population', 'unknown')
        }
        
        enhanced_cn_results.append(enhanced_result)
    
    cn_results_df = pd.DataFrame(enhanced_cn_results)
    
    # Estimate enhanced gene-level copy numbers
    print("Estimating enhanced gene-level copy numbers...")
    gene_results_df = estimate_enhanced_gene_copy_numbers(cn_results_df)
    
    # Save results
    cn_results_df.to_csv(args.output_file, index=False, sep='\t')
    
    gene_output_file = args.output_file.replace('.txt', '_enhanced_gene_level.txt')
    gene_results_df.to_csv(gene_output_file, index=False, sep='\t')
    
    # Save enhanced thresholds and quality metrics
    threshold_output_file = args.output_file.replace('.txt', '_enhanced_thresholds.txt')
    with open(threshold_output_file, 'w') as f:
        f.write("# Enhanced copy number thresholds and quality metrics used\n")
        f.write("parameter\tvalue\tdescription\n")
        f.write(f"homozygous_deletion\t{thresholds['homozygous_deletion']}\tCN=0 threshold\n")
        f.write(f"heterozygous_deletion\t{thresholds['heterozygous_deletion']}\tCN=1 threshold\n")
        f.write(f"normal_lower\t{thresholds['normal_lower']}\tCN=2 lower bound\n")
        f.write(f"normal_upper\t{thresholds['normal_upper']}\tCN=2 upper bound\n")
        f.write(f"duplication\t{thresholds['duplication']}\tCN=3 threshold\n")
        f.write(f"high_confidence\t{thresholds['high_confidence']}\tHigh confidence distance\n")
        f.write(f"medium_confidence\t{thresholds['medium_confidence']}\tMedium confidence distance\n")
        f.write(f"min_coverage\t{QUALITY_THRESHOLDS['min_coverage']}\tMinimum coverage threshold\n")
        f.write(f"max_missing_rate\t{QUALITY_THRESHOLDS['max_missing_rate']}\tMaximum missing data rate\n")
        f.write(f"min_segment_size\t{QUALITY_THRESHOLDS['min_segment_size']}\tMinimum segment size\n")
    
    # Create enhanced plots
    if not args.no_plots:
        try:
            create_enhanced_cnv_visualization(cn_results_df, gene_results_df, output_dir, thresholds)
        except Exception as e:
            print(f"Warning: Could not create enhanced plots: {e}")
    
    # Print comprehensive summary
    print(f"\nEnhanced copy number estimation completed!")
    print(f"Exon-level results saved to: {args.output_file}")
    print(f"Gene-level results saved to: {gene_output_file}")
    print(f"Enhanced parameters saved to: {threshold_output_file}")
    
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
    
    print(f"\nConfidence score statistics:")
    conf_stats = cn_results_df['confidence_score'].describe()
    print(conf_stats)
    
    print(f"\nSegmentation method performance:")
    seg_conf = cn_results_df.groupby(['segmentation_method', 'confidence']).size().unstack(fill_value=0)
    print(seg_conf)
    
    print(f"\nClinical significance summary (gene-level):")
    if not gene_results_df.empty:
        clinical_summary = gene_results_df.groupby(['gene', 'clinical_significance']).size().unstack(fill_value=0)
        print(clinical_summary)
    
    # Quality assessment
    print(f"\nQuality assessment:")
    low_coverage = (cn_results_df['raw_coverage'] < QUALITY_THRESHOLDS['min_coverage']).sum()
    high_missing = (cn_results_df.groupby('sample_id').apply(
        lambda x: (x[['z_score', 'robust_z_score']].isna().all(axis=1).mean() > QUALITY_THRESHOLDS['max_missing_rate']).any()
    )).sum()
    
    print(f"  Low coverage calls: {low_coverage} ({100*low_coverage/len(cn_results_df):.1f}%)")
    print(f"  High missing data samples: {high_missing}")
    print(f"  Mean confidence score: {cn_results_df['confidence_score'].mean():.3f}")
    print(f"  High confidence calls: {(cn_results_df['confidence'] == 'high').sum()} ({100*(cn_results_df['confidence'] == 'high').sum()/len(cn_results_df):.1f}%)")

if __name__ == "__main__":
    main()
