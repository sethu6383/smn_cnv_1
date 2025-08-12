#!/usr/bin/env python3

"""
generate_report.py - Generate comprehensive per-sample reports for SMN CNV analysis
Usage: python generate_report.py <cn_results_file> <allele_counts_file> <output_dir> [--sample sample_id]
"""

import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

def load_analysis_data(cn_file, allele_file):
    """Load copy number and allele count analysis results."""
    try:
        cn_df = pd.read_csv(cn_file, sep='\t')
        print(f"Loaded copy number data: {len(cn_df)} records")
    except Exception as e:
        print(f"Error loading copy number file: {e}")
        return None, None
    
    try:
        allele_df = pd.read_csv(allele_file, sep='\t')
        print(f"Loaded allele count data: {len(allele_df)} records")
    except Exception as e:
        print(f"Warning: Could not load allele count file: {e}")
        allele_df = pd.DataFrame()
    
    return cn_df, allele_df

def create_sample_summary(sample_id, cn_df, allele_df):
    """Create a summary for a specific sample."""
    sample_cn = cn_df[cn_df['sample_id'] == sample_id]
    sample_allele = allele_df[allele_df['sample_id'] == sample_id] if not allele_df.empty else pd.DataFrame()
    
    if sample_cn.empty:
        return None
    
    summary = {
        'sample_id': sample_id,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sample_type': sample_cn['sample_type'].iloc[0] if 'sample_type' in sample_cn.columns else 'unknown',
        'population': sample_cn['population'].iloc[0] if 'population' in sample_cn.columns else 'unknown'
    }
    
    # Exon-level results
    exon_results = {}
    for _, row in sample_cn.iterrows():
        exon_results[row['exon']] = {
            'z_score': row['z_score'],
            'copy_number': row['copy_number'],
            'cn_category': row['cn_category'],
            'confidence': row['confidence'],
            'raw_coverage': row.get('raw_coverage', np.nan)
        }
    
    summary['exon_results'] = exon_results
    
    # Gene-level estimates (infer from exon data)
    smn1_exons = [k for k in exon_results.keys() if 'SMN1' in k]
    smn2_exons = [k for k in exon_results.keys() if 'SMN2' in k]
    
    gene_results = {}
    for gene, exons in [('SMN1', smn1_exons), ('SMN2', smn2_exons)]:
        if exons:
            cn_values = [exon_results[exon]['copy_number'] for exon in exons if not pd.isna(exon_results[exon]['copy_number'])]
            z_scores = [exon_results[exon]['z_score'] for exon in exons if not pd.isna(exon_results[exon]['z_score'])]
            
            if cn_values:
                gene_results[gene] = {
                    'estimated_copy_number': np.median(cn_values),
                    'mean_z_score': np.mean(z_scores) if z_scores else np.nan,
                    'exon_consistency': len(set(cn_values)) == 1,
                    'n_exons': len(exons)
                }
    
    summary['gene_results'] = gene_results
    
    # Allele count results
    allele_results = {}
    if not sample_allele.empty:
        for _, row in sample_allele.iterrows():
            snp_key = f"{row['snp_name']}_{row['gene']}"
            allele_results[snp_key] = {
                'ref_count': row['ref_count'],
                'alt_count': row['alt_count'],
                'total_depth': row['total_depth'],
                'ref_freq': row['ref_freq'],
                'alt_freq': row['alt_freq'],
                'position': f"{row['chrom']}:{row['pos']}"
            }
    
    summary['allele_results'] = allele_results
    
    # Clinical interpretation
    summary['clinical_interpretation'] = interpret_results(summary)
    
    return summary

def interpret_results(summary):
    """Provide clinical interpretation of the results."""
    interpretation = {
        'overall_risk': 'unknown',
        'sma_risk_category': 'unknown',
        'carrier_status': 'unknown',
        'notes': []
    }
    
    gene_results = summary.get('gene_results', {})
    
    smn1_cn = gene_results.get('SMN1', {}).get('estimated_copy_number', np.nan)
    smn2_cn = gene_results.get('SMN2', {}).get('estimated_copy_number', np.nan)
    
    # SMN1 interpretation
    if not pd.isna(smn1_cn):
        if smn1_cn == 0:
            interpretation['sma_risk_category'] = 'affected'
            interpretation['overall_risk'] = 'high'
            interpretation['notes'].append('SMN1 homozygous deletion detected - likely SMA affected')
        elif smn1_cn == 1:
            interpretation['sma_risk_category'] = 'carrier'
            interpretation['overall_risk'] = 'medium'
            interpretation['carrier_status'] = 'SMN1_carrier'
            interpretation['notes'].append('SMN1 heterozygous deletion - SMA carrier')
        elif smn1_cn == 2:
            interpretation['sma_risk_category'] = 'normal'
            interpretation['overall_risk'] = 'low'
            interpretation['notes'].append('Normal SMN1 copy number')
        elif smn1_cn >= 3:
            interpretation['sma_risk_category'] = 'duplication'
            interpretation['overall_risk'] = 'low'
            interpretation['notes'].append('SMN1 duplication detected')
    
    # SMN2 interpretation
    if not pd.isna(smn2_cn):
        if smn2_cn == 0:
            interpretation['notes'].append('SMN2 homozygous deletion detected')
        elif smn2_cn == 1:
            interpretation['notes'].append('SMN2 heterozygous deletion detected')
        elif smn2_cn >= 3:
            interpretation['notes'].append('SMN2 duplication detected - may modify SMA severity')
    
    # Combined interpretation
    if not pd.isna(smn1_cn) and not pd.isna(smn2_cn):
        total_smn = smn1_cn + smn2_cn
        interpretation['total_smn_copies'] = total_smn
        
        if smn1_cn == 0 and smn2_cn >= 2:
            interpretation['notes'].append('SMN1 deletion with normal SMN2 - phenotype depends on SMN2 expression')
    
    return interpretation

def create_sample_plot(sample_summary, output_file):
    """Create a visualization for the sample results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"SMN CNV Analysis Report - {sample_summary['sample_id']}", fontsize=16, fontweight='bold')
    
    # Plot 1: Z-scores by exon
    ax1 = axes[0, 0]
    exon_results = sample_summary['exon_results']
    exons = list(exon_results.keys())
    z_scores = [exon_results[exon]['z_score'] for exon in exons]
    
    colors = []
    for exon in exons:
        cn = exon_results[exon]['copy_number']
        if cn == 0:
            colors.append('red')
        elif cn == 1:
            colors.append('orange')
        elif cn == 2:
            colors.append('green')
        elif cn == 3:
            colors.append('blue')
        else:
            colors.append('purple')
    
    bars = ax1.bar(exons, z_scores, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.axhline(y=-1.5, color='red', linestyle='--', alpha=0.5, label='CN threshold')
    ax1.axhline(y=1.5, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('Z-scores by Exon')
    ax1.set_ylabel('Z-score')
    ax1.set_xlabel('Exon')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    
    # Add copy number labels on bars
    for bar, exon in zip(bars, exons):
        height = bar.get_height()
        cn = exon_results[exon]['copy_number']
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height >= 0 else height - 0.3,
                f'CN={cn}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # Plot 2: Copy numbers by gene
    ax2 = axes[0, 1]
    gene_results = sample_summary['gene_results']
    if gene_results:
        genes = list(gene_results.keys())
        copy_numbers = [gene_results[gene]['estimated_copy_number'] for gene in genes]
        
        gene_colors = ['lightblue' if 'SMN1' in gene else 'lightcoral' for gene in genes]
        bars = ax2.bar(genes, copy_numbers, color=gene_colors, alpha=0.7)
        
        ax2.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Normal (CN=2)')
        ax2.set_title('Gene-level Copy Number Estimates')
        ax2.set_ylabel('Copy Number')
        ax2.set_xlabel('Gene')
        ax2.set_ylim(0, max(copy_numbers) + 0.5 if copy_numbers else 4)
        ax2.legend()
        
        # Add labels on bars
        for bar, cn in zip(bars, copy_numbers):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{cn:.1f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No gene-level data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Gene-level Copy Number Estimates')
    
    # Plot 3: Allele frequencies
    ax3 = axes[1, 0]
    allele_results = sample_summary['allele_results']
    if allele_results:
        snps = list(allele_results.keys())
        ref_freqs = [allele_results[snp]['ref_freq'] for snp in snps]
        alt_freqs = [allele_results[snp]['alt_freq'] for snp in snps]
        
        x = range(len(snps))
        width = 0.35
        
        ax3.bar([i - width/2 for i in x], ref_freqs, width, label='Reference allele', alpha=0.7)
        ax3.bar([i + width/2 for i in x], alt_freqs, width, label='Alternative allele', alpha=0.7)
        
        ax3.set_title('Allele Frequencies at Discriminating SNPs')
        ax3.set_ylabel('Allele Frequency')
        ax3.set_xlabel('SNP')
        ax3.set_xticks(x)
        ax3.set_xticklabels([snp.replace('_', '\n') for snp in snps], fontsize=8)
        ax3.legend()
        ax3.set_ylim(0, 1)
    else:
        ax3.text(0.5, 0.5, 'No allele count data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Allele Frequencies at Discriminating SNPs')
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    summary_text = f"Sample: {sample_summary['sample_id']}\n"
    summary_text += f"Type: {sample_summary['sample_type']}\n"
    summary_text += f"Population: {sample_summary['population']}\n"
    summary_text += f"Analysis Date: {sample_summary['analysis_date']}\n\n"
    
    interpretation = sample_summary['clinical_interpretation']
    summary_text += "Clinical Interpretation:\n"
    summary_text += f"Risk Category: {interpretation['sma_risk_category']}\n"
    summary_text += f"Overall Risk: {interpretation['overall_risk']}\n"
    summary_text += f"Carrier Status: {interpretation['carrier_status']}\n\n"
    
    summary_text += "Notes:\n"
    for note in interpretation['notes']:
        summary_text += f"• {note}\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def generate_html_report(sample_summary, output_file):
    """Generate an HTML report for the sample."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SMN CNV Analysis Report - {sample_summary['sample_id']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .risk-high {{ background-color: #ffebee; }}
            .risk-medium {{ background-color: #fff3e0; }}
            .risk-low {{ background-color: #e8f5e8; }}
            .risk-unknown {{ background-color: #f5f5f5; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .cn-0 {{ background-color: #ffcdd2; }}
            .cn-1 {{ background-color: #ffe0b2; }}
            .cn-2 {{ background-color: #c8e6c9; }}
            .cn-3 {{ background-color: #bbdefb; }}
            .cn-4 {{ background-color: #e1bee7; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>SMN CNV Analysis Report</h1>
            <h2>Sample: {sample_summary['sample_id']}</h2>
            <p><strong>Sample Type:</strong> {sample_summary['sample_type']}</p>
            <p><strong>Population:</strong> {sample_summary['population']}</p>
            <p><strong>Analysis Date:</strong> {sample_summary['analysis_date']}</p>
        </div>
    """
    
    # Clinical interpretation section
    interpretation = sample_summary['clinical_interpretation']
    risk_class = f"risk-{interpretation['overall_risk']}"
    
    html_content += f"""
        <div class="section {risk_class}">
            <h3>Clinical Interpretation</h3>
            <p><strong>SMA Risk Category:</strong> {interpretation['sma_risk_category']}</p>
            <p><strong>Overall Risk:</strong> {interpretation['overall_risk']}</p>
            <p><strong>Carrier Status:</strong> {interpretation['carrier_status']}</p>
            <h4>Clinical Notes:</h4>
            <ul>
    """
    
    for note in interpretation['notes']:
        html_content += f"<li>{note}</li>"
    
    html_content += """
            </ul>
        </div>
    """
    
    # Gene-level results
    html_content += """
        <div class="section">
            <h3>Gene-level Copy Number Estimates</h3>
            <table>
                <tr>
                    <th>Gene</th>
                    <th>Estimated Copy Number</th>
                    <th>Mean Z-score</th>
                    <th>Exon Consistency</th>
                    <th>Number of Exons</th>
                </tr>
    """
    
    gene_results = sample_summary['gene_results']
    for gene, data in gene_results.items():
        cn = data['estimated_copy_number']
        cn_class = f"cn-{int(cn) if not pd.isna(cn) else 'unknown'}"
        consistency = "Yes" if data['exon_consistency'] else "No"
        
        html_content += f"""
                <tr class="{cn_class}">
                    <td>{gene}</td>
                    <td>{cn:.1f}</td>
                    <td>{data['mean_z_score']:.2f}</td>
                    <td>{consistency}</td>
                    <td>{data['n_exons']}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    """
    
    # Exon-level results
    html_content += """
        <div class="section">
            <h3>Exon-level Results</h3>
            <table>
                <tr>
                    <th>Exon</th>
                    <th>Z-score</th>
                    <th>Copy Number</th>
                    <th>Category</th>
                    <th>Confidence</th>
                    <th>Raw Coverage</th>
                </tr>
    """
    
    exon_results = sample_summary['exon_results']
    for exon, data in exon_results.items():
        cn = data['copy_number']
        cn_class = f"cn-{int(cn) if not pd.isna(cn) else 'unknown'}"
        
        html_content += f"""
                <tr class="{cn_class}">
                    <td>{exon}</td>
                    <td>{data['z_score']:.2f}</td>
                    <td>{cn}</td>
                    <td>{data['cn_category']}</td>
                    <td>{data['confidence']}</td>
                    <td>{data['raw_coverage']:.1f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    """
    
    # Allele count results
    allele_results = sample_summary['allele_results']
    if allele_results:
        html_content += """
            <div class="section">
                <h3>Allele Count Results</h3>
                <table>
                    <tr>
                        <th>SNP</th>
                        <th>Position</th>
                        <th>Ref Count</th>
                        <th>Alt Count</th>
                        <th>Total Depth</th>
                        <th>Ref Frequency</th>
                        <th>Alt Frequency</th>
                    </tr>
        """
        
        for snp, data in allele_results.items():
            html_content += f"""
                    <tr>
                        <td>{snp}</td>
                        <td>{data['position']}</td>
                        <td>{data['ref_count']}</td>
                        <td>{data['alt_count']}</td>
                        <td>{data['total_depth']}</td>
                        <td>{data['ref_freq']:.3f}</td>
                        <td>{data['alt_freq']:.3f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Generate SMN CNV analysis reports')
    parser.add_argument('cn_results_file', help='Copy number results file')
    parser.add_argument('allele_counts_file', help='Allele counts file')
    parser.add_argument('output_dir', help='Output directory for reports')
    parser.add_argument('--sample', help='Generate report for specific sample only')
    parser.add_argument('--format', choices=['json', 'html', 'plot', 'all'], 
                       default='all', help='Output format(s)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    cn_df, allele_df = load_analysis_data(args.cn_results_file, args.allele_counts_file)
    
    if cn_df is None:
        print("Error: Could not load copy number data")
        sys.exit(1)
    
    # Get samples to process
    if args.sample:
        if args.sample not in cn_df['sample_id'].values:
            print(f"Error: Sample {args.sample} not found in data")
            sys.exit(1)
        samples_to_process = [args.sample]
    else:
        samples_to_process = cn_df['sample_id'].unique()
    
    print(f"Generating reports for {len(samples_to_process)} samples...")
    
    # Generate reports for each sample
    for sample_id in samples_to_process:
        print(f"Processing sample: {sample_id}")
        
        sample_summary = create_sample_summary(sample_id, cn_df, allele_df)
        
        if sample_summary is None:
            print(f"Warning: No data found for sample {sample_id}")
            continue
        
        # Generate outputs based on format
        sample_output_dir = output_dir / sample_id
        sample_output_dir.mkdir(exist_ok=True)
        
        if args.format in ['json', 'all']:
            json_file = sample_output_dir / f"{sample_id}_report.json"
            with open(json_file, 'w') as f:
                json.dump(sample_summary, f, indent=2, default=str)
            print(f"  JSON report: {json_file}")
        
        if args.format in ['html', 'all']:
            html_file = sample_output_dir / f"{sample_id}_report.html"
            generate_html_report(sample_summary, html_file)
            print(f"  HTML report: {html_file}")
        
        if args.format in ['plot', 'all']:
            plot_file = sample_output_dir / f"{sample_id}_plot.png"
            create_sample_plot(sample_summary, plot_file)
            print(f"  Plot: {plot_file}")
    
    print("Report generation completed!")

if __name__ == "__main__":
    main()
