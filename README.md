# SMN CNV Detection Pipeline

A comprehensive MVP pipeline for detecting copy number variations (CNVs) in SMN1 and SMN2 genes—specifically in exons 7 and 8—using whole exome sequencing (WES) data from indexed BAM files.

## Overview

This pipeline processes BAM files from a cohort of samples through a series of modules to detect SMN1/SMN2 copy number variations, which are crucial for Spinal Muscular Atrophy (SMA) diagnosis and carrier screening.

### Key Features

- **Automated BAM file discovery** from input directory
- **Smart sample type detection** based on filename patterns
- **Automated depth extraction** using samtools depth
- **Coverage normalization** using reference samples to compute Z-scores
- **Allele-specific counting** at known SMN1/SMN2-discriminating SNPs
- **Copy number estimation** using predefined thresholds
- **Comprehensive reporting** with HTML, JSON, and visual outputs
- **Clinical interpretation** for SMA risk assessment

## Pipeline Workflow

1. **BAM File Discovery**: Automatically find all BAM files in input directory
2. **Sample Type Detection**: Auto-classify samples as reference or test based on filenames
3. **Depth Extraction**: Extract read depth per exon using `samtools depth`
4. **Coverage Calculation**: Calculate average coverage per exon
5. **Allele Counting**: Perform allele-specific base counting at discriminating SNPs
6. **Normalization**: Normalize coverage using reference samples and compute Z-scores
7. **Copy Number Estimation**: Estimate CN states using predefined thresholds
8. **Report Generation**: Create per-sample reports with clinical interpretation

## Copy Number Thresholds

The pipeline uses the following Z-score thresholds for copy number estimation:

- **CN=0** (Homozygous deletion): Z-score ≤ -2.5
- **CN=1** (Heterozygous deletion): Z-score -2.5 to -1.5
- **CN=2** (Normal): Z-score -1.5 to +1.5
- **CN=3** (Duplication): Z-score +1.5 to +2.5
- **CN=4+** (Multi-duplication): Z-score > +2.5

## Installation and Requirements

### System Requirements

- Linux/Unix environment
- samtools (≥1.10)
- Python 3.7+

### Python Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Installation

1. Clone or download the pipeline:
```bash
# If using git
git clone <repository_url>
cd smn_cnv_pipeline

# Or extract from archive
tar -xzf smn_cnv_pipeline.tar.gz
cd smn_cnv_pipeline
```

2. Make scripts executable:
```bash
chmod +x run_pipeline.sh bin/*.sh
```

## Configuration

### Input Data Preparation

1. **Organize BAM Files**: Place all BAM files in a single directory
   ```
   /path/to/bam/files/
   ├── ref001.bam
   ├── ref001.bam.bai
   ├── ref002.bam
   ├── ref002.bam.bai
   ├── control_sample.bam
   ├── control_sample.bam.bai
   ├── patient001.bam
   ├── patient001.bam.bai
   └── test_sample.bam
       test_sample.bam.bai
   ```

2. **Ensure BAM Indexing**: All BAM files must have corresponding .bai index files
   ```bash
   samtools index your_file.bam
   ```

### Sample Type Auto-Detection

The pipeline automatically classifies samples based on filename patterns:

- **Reference samples**: Filenames containing `ref`, `control`, or `normal`
  - Examples: `ref001.bam`, `control_sample.bam`, `normal_01.bam`
- **Test samples**: All other BAM files
  - Examples: `patient001.bam`, `sample_xyz.bam`, `test001.bam`

You can override auto-detection using the `--sample-type` option.

### Genomic Coordinates

The pipeline includes pre-configured files for GRCh38:
- `config/smn_exons.bed`: SMN1/SMN2 exon 7 and 8 coordinates
- `config/discriminating_snps.txt`: Known SMN1/SMN2 discriminating SNPs

These files are ready to use but can be modified if needed.

## Usage

### Basic Usage

```bash
# Auto-detect sample types from filenames
./run_pipeline.sh /path/to/bam/files/
```

### Advanced Usage

```bash
# All samples are reference samples
./run_pipeline.sh /path/to/bam/files/ --sample-type reference

# All samples are test samples  
./run_pipeline.sh /path/to/bam/files/ --sample-type test

# Custom output directory
./run_pipeline.sh /path/to/bam/files/ --results /custom/output/dir

# Fast analysis without plots
./run_pipeline.sh /path/to/bam/files/ --skip-plots
```

### Command Line Options

- `input_bam_dir`: **Required** - Directory containing BAM files to analyze
- `--config DIR`: Configuration directory (default: ./config)
- `--results DIR`: Results directory (default: ./results)
- `--sample-type TYPE`: Sample type: `reference`, `test`, or `auto` (default: auto)
- `--skip-plots`: Skip generating plots to speed up analysis
- `--verbose`: Enable verbose output
- `--help`: Show help message

## Output Structure

```
results/
├── depth/                    # Read depth files
│   ├── SAMPLE001_depth.txt
│   ├── coverage_summary.txt
│   └── coverage_summary_pivot.txt
├── allele_counts/            # Allele counting results
│   ├── allele_counts.txt
│   ├── allele_counts_summary.txt
│   └── sample_info.txt
├── normalized/               # Normalized data and Z-scores
│   ├── z_scores.txt
│   ├── z_scores_ref_stats.txt
│   └── plots/
├── cnv_calls/               # Copy number estimates
│   ├── copy_numbers.txt
│   ├── copy_numbers_gene_level.txt
│   ├── copy_numbers_thresholds.txt
│   └── plots/
├── reports/                 # Per-sample reports
│   ├── SAMPLE001/
│   │   ├── SAMPLE001_report.html
│   │   ├── SAMPLE001_report.json
│   │   └── SAMPLE001_plot.png
│   └── ...
└── pipeline_summary.txt    # Overall pipeline summary
```

## Interpreting Results

### Clinical Significance

- **SMN1 CN=0**: Likely SMA affected (homozygous deletion)
- **SMN1 CN=1**: SMA carrier (heterozygous deletion)
- **SMN1 CN=2**: Normal copy number
- **SMN1 CN≥3**: Gene duplication

### Key Output Files

1. **`reports/SAMPLE_ID/SAMPLE_ID_report.html`**: Comprehensive HTML report with clinical interpretation
2. **`cnv_calls/copy_numbers_gene_level.txt`**: Gene-level copy number estimates
3. **`normalized/z_scores.txt`**: Detailed Z-scores for all samples and exons

### Quality Control

- Check `logs/` directory for any errors or warnings
- Verify reference sample count (≥3 recommended)
- Review Z-score distributions in plots
- Check exon consistency within genes

## Troubleshooting

### Common Issues

1. **"No BAM files found in directory"**
   - Verify BAM files are in the specified directory
   - Check file permissions

2. **"BAM index not found"**
   - Index BAM files: `samtools index file.bam`
   - Ensure .bai files are in same directory as BAM files

3. **"Very few reference samples"**
   - Use more descriptive filenames for reference samples
   - Manually specify sample type: `--sample-type reference`
   - Include at least 3-5 reference samples for reliable normalization

4. **Python package errors**
   - Install missing packages: `pip install pandas numpy matplotlib seaborn scipy`
   - Check Python version (≥3.7 required)

### Log Files

All operations are logged in `logs/` directory:
- `depth_extraction.log`
- `coverage_calculation.log`
- `allele_counting.log`
- `normalization.log`
- `copy_number_estimation.log`
- `report_generation.log`

## Performance Considerations

- **Runtime**: ~10-30 minutes for 10-50 samples
- **Memory**: ~1-4 GB depending on sample count
- **Storage**: ~100-500 MB per sample for intermediate files

### Optimization Tips

- Use `--skip-plots` for faster analysis when plots aren't needed
- Process samples in batches if memory is limited
- Consider using SSD storage for better I/O performance

## Validation and Accuracy

### Expected Performance

- **Sensitivity**: >95% for detecting CN=0 and CN=1 variants
- **Specificity**: >98% for normal samples (CN=2)
- **Reproducibility**: CV < 5% for technical replicates

### Quality Metrics

The pipeline provides several quality metrics:
- Coverage depth per exon
- Z-score distributions
- Reference sample statistics
- Confidence scores for copy number calls

## Clinical Considerations

### SMA and SMN Genes

- **SMN1**: Primary functional gene, deletions cause SMA
- **SMN2**: Pseudogene with reduced function, can partially compensate
- **Carrier frequency**: ~1 in 50 in most populations
- **Disease frequency**: ~1 in 10,000 births

### Limitations

- Pipeline designed for exons 7 and 8 only
- Requires adequate coverage (≥20x recommended)
- Cannot detect point mutations or small indels
- Results require clinical correlation

## Examples

### Quick Start Example

```bash
# 1. Prepare your data
mkdir -p /data/sma_analysis/bams
# Copy your BAM files to this directory

# 2. Index BAM files if needed
for bam in /data/sma_analysis/bams/*.bam; do
    samtools index "$bam"
done

# 3. Run pipeline
./run_pipeline.sh /data/sma_analysis/bams/

# 4. View results
open results/pipeline_summary.txt
open results/reports/*/report.html
```

### Different Sample Type Scenarios

```bash
# Scenario 1: Mixed samples (auto-detection)
./run_pipeline.sh /data/mixed_samples/
# Files named ref*.bam, control*.bam → reference
# Other files → test

# Scenario 2: All reference samples (e.g., building reference database)
./run_pipeline.sh /data/reference_cohort/ --sample-type reference

# Scenario 3: All test samples (with external reference data)
./run_pipeline.sh /data/patient_samples/ --sample-type test
```

## Support and Contributing

### Getting Help

1. Check the troubleshooting section
2. Review log files for specific errors
3. Verify BAM file organization and indexing

### Contributing

To contribute improvements or report issues:
1. Document the problem with log files
2. Include system information and dependency versions
3. Provide example data structure (anonymized)

## License and Citation

Please cite this pipeline in publications:

```
SMN CNV Detection Pipeline: A comprehensive tool for detecting copy number 
variations in SMN1 and SMN2 genes from whole exome sequencing data.
```

## Version History

- **v2.0**: Updated to use input directory instead of manifest file
  - Automatic BAM file discovery
  - Smart sample type detection
  - Simplified workflow
- **v1.0**: Initial MVP release with core functionality
  - Depth extraction and coverage normalization
  - Z-score based copy number estimation
  - HTML report generation
  - Support for GRCh38 coordinates

---

For technical support or questions, please refer to the troubleshooting section or check the log files for detailed error information.
