#!/usr/bin/env python3

"""
calculate_coverage.py - Calculate average coverage per exon from depth files
Usage: python calculate_coverage.py <depth_dir> <bed_file> <output_file>
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

def read_bed_file(bed_file):
    """Read BED file and return exon coordinates."""
    exons = {}
    with open(bed_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 4:
                chrom, start, end, exon_name = parts[:4]
                exons[exon_name] = {
                    'chrom': chrom,
                    'start': int(start),
                    'end': int(end),
                    'length': int(end) - int(start)
                }
    return exons

def calculate_exon_coverage(depth_file, exons):
    """Calculate average coverage for each exon from depth file."""
    coverage_data = {}
    
    # Initialize coverage data for each exon
    for exon_name in exons:
        coverage_data[exon_name] = {'total_depth': 0, 'positions': 0, 'avg_coverage': 0}
    
    # Read depth file
    try:
        with open(depth_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    chrom, pos, depth = parts[0], int(parts[1]), int(parts[2])
                    
                    # Find which exon this position belongs to
                    for exon_name, exon_info in exons.items():
                        if (chrom == exon_info['chrom'] and 
                            exon_info['start'] <= pos <= exon_info['end']):
                            coverage_data[exon_name]['total_depth'] += depth
                            coverage_data[exon_name]['positions'] += 1
                            break
    except FileNotFoundError:
        print(f"Warning: Depth file not found: {depth_file}")
        return coverage_data
    
    # Calculate average coverage
    for exon_name in coverage_data:
        if coverage_data[exon_name]['positions'] > 0:
            coverage_data[exon_name]['avg_coverage'] = (
                coverage_data[exon_name]['total_depth'] / 
                coverage_data[exon_name]['positions']
            )
        else:
            coverage_data[exon_name]['avg_coverage'] = 0
    
    return coverage_data

def main():
    if len(sys.argv) != 4:
        print("Usage: python calculate_coverage.py <depth_dir> <bed_file> <output_file>")
        print("  depth_dir: Directory containing depth files")
        print("  bed_file: BED file with exon coordinates")
        print("  output_file: Output file for coverage summary")
        sys.exit(1)
    
    depth_dir = sys.argv[1]
    bed_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Read exon coordinates
    print("Reading exon coordinates...")
    exons = read_bed_file(bed_file)
    print(f"Found {len(exons)} exons")
    
    # Process all depth files
    results = []
    depth_files = Path(depth_dir).glob("*_depth.txt")
    
    for depth_file in depth_files:
        sample_id = depth_file.stem.replace('_depth', '')
        print(f"Processing sample: {sample_id}")
        
        coverage_data = calculate_exon_coverage(str(depth_file), exons)
        
        # Add to results
        for exon_name, data in coverage_data.items():
            results.append({
                'sample_id': sample_id,
                'exon': exon_name,
                'avg_coverage': data['avg_coverage'],
                'total_positions': data['positions'],
                'total_depth': data['total_depth']
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    
    if not df.empty:
        # Pivot table for easier analysis
        pivot_df = df.pivot(index='sample_id', columns='exon', values='avg_coverage')
        
        # Save both formats
        df.to_csv(output_file, index=False, sep='\t')
        pivot_file = output_file.replace('.txt', '_pivot.txt')
        pivot_df.to_csv(pivot_file, sep='\t')
        
        print(f"Coverage data saved to: {output_file}")
        print(f"Pivot table saved to: {pivot_file}")
        print(f"Processed {len(df['sample_id'].unique())} samples")
        print(f"Coverage summary:")
        print(pivot_df.describe())
    else:
        print("No coverage data found!")

if __name__ == "__main__":
    main()
