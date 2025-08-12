#!/usr/bin/env python3

"""
allele_count.py - Perform allele-specific counting at SMN1/SMN2 discriminating SNPs
Usage: python allele_count.py <input_bam_dir> <snp_file> <output_dir> [--sample-type TYPE]
"""

import sys
import os
import subprocess
import pandas as pd
from pathlib import Path
import re
import argparse
import glob

def read_snp_file(snp_file):
    """Read SNP configuration file."""
    snps = []
    with open(snp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 6:
                chrom, pos, ref, alt, snp_name, gene = parts[:6]
                snps.append({
                    'chrom': chrom,
                    'pos': int(pos),
                    'ref': ref,
                    'alt': alt,
                    'snp_name': snp_name,
                    'gene': gene
                })
    return snps

def find_bam_files(input_dir):
    """Find all BAM files in the input directory."""
    bam_pattern = os.path.join(input_dir, "*.bam")
    bam_files = glob.glob(bam_pattern)
    
    if not bam_files:
        print(f"No BAM files found in directory: {input_dir}")
        return []
    
    # Create sample information
    samples = []
    for bam_path in bam_files:
        sample_id = os.path.basename(bam_path).replace('.bam', '')
        
        # Auto-detect sample type based on filename
        if any(keyword in sample_id.lower() for keyword in ['ref', 'control', 'normal']):
            sample_type = 'reference'
        else:
            sample_type = 'test'
        
        samples.append({
            'sample_id': sample_id,
            'bam_path': bam_path,
            'sample_type': sample_type
        })
    
    return samples

def parse_pileup_line(line):
    """Parse a single line from samtools mpileup output."""
    parts = line.strip().split('\t')
    if len(parts) < 5:
        return None
    
    chrom, pos, ref_base, depth, bases = parts[:5]
    pos = int(pos)
    depth = int(depth)
    
    if depth == 0:
        return {
            'chrom': chrom,
            'pos': pos,
            'ref_base': ref_base,
            'depth': depth,
            'ref_count': 0,
            'alt_count': 0,
            'other_count': 0
        }
    
    # Clean up the base string
    # Remove start/end markers (^ and $)
    bases = re.sub(r'\^.', '', bases)  # Remove ^quality
    bases = re.sub(r'\$', '', bases)   # Remove $
    
    # Count bases
    ref_count = bases.count('.') + bases.count(',')  # Matches reference
    
    # Count specific bases (case insensitive)
    base_counts = {}
    for base in 'ATCG':
        base_counts[base] = bases.upper().count(base)
    
    return {
        'chrom': chrom,
        'pos': pos,
        'ref_base': ref_base.upper(),
        'depth': depth,
        'ref_count': ref_count,
        'base_counts': base_counts,
        'raw_bases': bases
    }

def count_alleles_at_position(bam_file, chrom, pos, ref_allele, alt_allele):
    """Count alleles at a specific position using samtools mpileup."""
    try:
        # Run samtools mpileup for this specific position
        cmd = [
            'samtools', 'mpileup', 
            '-r', f'{chrom}:{pos}-{pos}',
            '-Q', '20',  # Minimum base quality
            '-q', '20',  # Minimum mapping quality
            bam_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if not result.stdout.strip():
            return {
                'ref_count': 0,
                'alt_count': 0,
                'total_depth': 0,
                'other_count': 0,
                'raw_output': ''
            }
        
        # Parse the output
        pileup_data = parse_pileup_line(result.stdout.strip())
        
        if pileup_data is None:
            return {
                'ref_count': 0,
                'alt_count': 0,
                'total_depth': 0,
                'other_count': 0,
                'raw_output': result.stdout.strip()
            }
        
        # Count reference and alternative alleles
        ref_count = pileup_data['ref_count']
        alt_count = pileup_data['base_counts'].get(alt_allele.upper(), 0)
        
        # Calculate other bases
        total_called = ref_count + sum(pileup_data['base_counts'].values())
        other_count = max(0, pileup_data['depth'] - total_called)
        
        return {
            'ref_count': ref_count,
            'alt_count': alt_count,
            'total_depth': pileup_data['depth'],
            'other_count': other_count,
            'raw_output': result.stdout.strip(),
            'base_counts': pileup_data['base_counts']
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error running samtools mpileup: {e}")
        return {
            'ref_count': 0,
            'alt_count': 0,
            'total_depth': 0,
            'other_count': 0,
            'raw_output': f'Error: {e}'
        }

def process_sample(sample_info, snps, output_dir):
    """Process a single sample for all SNPs."""
    results = []
    sample_id = sample_info['sample_id']
    bam_path = sample_info['bam_path']
    sample_type = sample_info['sample_type']
    
    print(f"Processing sample: {sample_id} (type: {sample_type})")
    
    # Check if BAM file exists
    if not os.path.exists(bam_path):
        print(f"Warning: BAM file not found: {bam_path}")
        return results
    
    for snp in snps:
        print(f"  Counting alleles at {snp['snp_name']} ({snp['chrom']}:{snp['pos']})")
        
        allele_counts = count_alleles_at_position(
            bam_path, snp['chrom'], snp['pos'], snp['ref'], snp['alt']
        )
        
        result = {
            'sample_id': sample_id,
            'chrom': snp['chrom'],
            'pos': snp['pos'],
            'snp_name': snp['snp_name'],
            'gene': snp['gene'],
            'ref_allele': snp['ref'],
            'alt_allele': snp['alt'],
            'ref_count': allele_counts['ref_count'],
            'alt_count': allele_counts['alt_count'],
            'total_depth': allele_counts['total_depth'],
            'other_count': allele_counts['other_count'],
            'ref_freq': allele_counts['ref_count'] / max(1, allele_counts['total_depth']),
            'alt_freq': allele_counts['alt_count'] / max(1, allele_counts['total_depth']),
            'sample_type': sample_type
        }
        
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Perform allele-specific counting at SMN1/SMN2 discriminating SNPs')
    parser.add_argument('input_bam_dir', help='Directory containing BAM files')
    parser.add_argument('snp_file', help='Path to SNP configuration file')
    parser.add_argument('output_dir', help='Output directory for allele count files')
    parser.add_argument('--sample-type', choices=['reference', 'test', 'auto'], 
                       default='auto', help='Sample type for all samples (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read SNP configurations
    print("Reading SNP configurations...")
    snps = read_snp_file(args.snp_file)
    print(f"Found {len(snps)} SNPs to analyze")
    
    # Find BAM files
    print(f"Scanning for BAM files in: {args.input_bam_dir}")
    samples = find_bam_files(args.input_bam_dir)
    
    if not samples:
        print("Error: No BAM files found")
        sys.exit(1)
    
    # Override sample type if specified
    if args.sample_type != 'auto':
        for sample in samples:
            sample['sample_type'] = args.sample_type
    
    print(f"Found {len(samples)} samples to process")
    print("Sample types:")
    type_counts = {}
    for sample in samples:
        sample_type = sample['sample_type']
        type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
    for stype, count in type_counts.items():
        print(f"  {stype}: {count} samples")
    
    # Process all samples
    all_results = []
    for sample in samples:
        sample_results = process_sample(sample, snps, args.output_dir)
        all_results.extend(sample_results)
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = os.path.join(args.output_dir, 'allele_counts.txt')
        df.to_csv(output_file, index=False, sep='\t')
        
        # Create summary by SNP
        summary_file = os.path.join(args.output_dir, 'allele_counts_summary.txt')
        summary = df.groupby(['snp_name', 'gene']).agg({
            'ref_count': ['mean', 'std', 'min', 'max'],
            'alt_count': ['mean', 'std', 'min', 'max'],
            'total_depth': ['mean', 'std', 'min', 'max'],
            'ref_freq': ['mean', 'std'],
            'alt_freq': ['mean', 'std']
        }).round(3)
        
        summary.to_csv(summary_file, sep='\t')
        
        # Save sample info
        sample_info_file = os.path.join(args.output_dir, 'sample_info.txt')
        sample_df = pd.DataFrame(samples)
        sample_df.to_csv(sample_info_file, index=False, sep='\t')
        
        print(f"Allele counting completed!")
        print(f"Results saved to: {output_file}")
        print(f"Summary saved to: {summary_file}")
        print(f"Sample info saved to: {sample_info_file}")
        print(f"Processed {len(df['sample_id'].unique())} samples")
    else:
        print("No allele count data generated!")

if __name__ == "__main__":
    main()
