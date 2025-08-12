#!/bin/bash

# example_run.sh - Example of how to run the SMN CNV detection pipeline
# This is a demonstration script - modify paths for your actual data

echo "SMN CNV Detection Pipeline - Example Run"
echo "========================================"

# Set pipeline directory
PIPELINE_DIR="/workspace/smn_cnv_pipeline"
cd "$PIPELINE_DIR"

echo "1. Prepare your BAM files:"
echo "   - Place all BAM files in a single directory"
echo "   - Ensure BAM files are indexed (.bai files present)"
echo "   - Example directory structure:"
echo "   /data/bams/"
echo "   ├── ref001.bam"
echo "   ├── ref001.bam.bai"
echo "   ├── ref002.bam"
echo "   ├── ref002.bam.bai"
echo "   ├── test001.bam"
echo "   └── test001.bam.bai"
echo ""

echo "2. Create BAM indices if needed:"
echo "   samtools index /data/bams/ref001.bam"
echo "   samtools index /data/bams/ref002.bam"
echo "   samtools index /data/bams/test001.bam"
echo ""

echo "3. Run the pipeline:"
echo "   Basic usage (auto-detect sample types):"
echo "   ./run_pipeline.sh /data/bams/"
echo ""
echo "   All samples are reference samples:"
echo "   ./run_pipeline.sh /data/bams/ --sample-type reference"
echo ""
echo "   All samples are test samples:"
echo "   ./run_pipeline.sh /data/bams/ --sample-type test"
echo ""
echo "   Custom output directory:"
echo "   ./run_pipeline.sh /data/bams/ --results /custom/output/"
echo ""
echo "   Fast analysis without plots:"
echo "   ./run_pipeline.sh /data/bams/ --skip-plots"
echo ""

echo "4. Sample type auto-detection:"
echo "   - Files with 'ref', 'control', or 'normal' in name → reference samples"
echo "   - All other files → test samples"
echo "   - Examples:"
echo "     * ref001.bam → reference"
echo "     * control_sample.bam → reference"
echo "     * normal_01.bam → reference"
echo "     * patient001.bam → test"
echo "     * sample_xyz.bam → test"
echo ""

echo "5. Results will be available in:"
echo "   - results/reports/SAMPLE_ID/SAMPLE_ID_report.html (main reports)"
echo "   - results/cnv_calls/copy_numbers_gene_level.txt (summary)"
echo "   - results/pipeline_summary.txt (overall summary)"
echo ""

echo "6. For help:"
echo "   ./run_pipeline.sh --help"
echo ""

echo "Example with dummy data (for testing):"
echo "   # Create a test directory with some dummy BAM files"
echo "   mkdir -p /tmp/test_bams"
echo "   # Copy or link your BAM files here"
echo "   ./run_pipeline.sh /tmp/test_bams/"
echo ""

echo "Note: Replace '/data/bams/' with the actual path to your BAM files directory"
