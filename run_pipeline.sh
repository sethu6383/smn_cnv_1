#!/bin/bash

# run_pipeline.sh - Master script for SMN CNV detection pipeline
# Usage: ./run_pipeline.sh <input_bam_dir> [--config config_dir] [--results results_dir] [--sample-type TYPE] [--skip-plots]

set -euo pipefail

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="/data/SMN/cnv_pipeline/config"
RESULTS_DIR="$PIPELINE_DIR/data/SMN/results_CNV"
BIN_DIR="/data/SMN/cnv_pipeline/bin"
LOG_DIR="$PIPELINE_DIR/logs"

# Default configuration files
BED_FILE="/data/SMN/cnv_pipeline/config/smn_exons.bed"
SNP_FILE="/data/SMN/cnv_pipeline/config/discriminating_snps.txt"


# Pipeline options
SKIP_PLOTS=false
VERBOSE=false
SAMPLE_TYPE="auto"
INPUT_BAM_DIR="/data/SMN/BAM"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

print_error() {
    print_status "$RED" "ERROR: $1"
}

print_warning() {
    print_status "$YELLOW" "WARNING: $1"
}

print_info() {
    print_status "$BLUE" "INFO: $1"
}

print_success() {
    print_status "$GREEN" "SUCCESS: $1"
}

# Function to check if required tools are available
check_dependencies() {
    print_info "Checking dependencies..."
    
    local missing_tools=()
    
    # Check required command-line tools
    for tool in samtools python3; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    # Check Python packages
    if ! python3 -c "import pandas, numpy, matplotlib, seaborn, scipy" &> /dev/null; then
        print_warning "Some Python packages may be missing. Required: pandas, numpy, matplotlib, seaborn, scipy"
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    print_success "All dependencies found"
}

# Function to validate configuration files and input
validate_config() {
    print_info "Validating configuration files and input..."
    
    # Check if configuration files exist
    for file in "$BED_FILE" "$SNP_FILE"; do
        if [ ! -f "$file" ]; then
            print_error "Configuration file not found: $file"
            exit 1
        fi
    done
    
    # Validate BED file
    if ! grep -q "SMN1_exon" "$BED_FILE" || ! grep -q "SMN2_exon" "$BED_FILE"; then
        print_error "BED file does not contain expected SMN1/SMN2 exon entries"
        exit 1
    fi
    
    # Validate input BAM directory
    if [ ! -d "$INPUT_BAM_DIR" ]; then
        print_error "Input BAM directory not found: $INPUT_BAM_DIR"
        exit 1
    fi
    
    # Check for BAM files
    local bam_count=$(find "$INPUT_BAM_DIR" -name "*.bam" | wc -l)
    if [ "$bam_count" -eq 0 ]; then
        print_error "No BAM files found in directory: $INPUT_BAM_DIR"
        exit 1
    fi
    
    print_success "Configuration validated ($bam_count BAM files found)"
}

# Function to create output directories
setup_directories() {
    print_info "Setting up output directories..."
    
    mkdir -p "$RESULTS_DIR"/{depth,allele_counts,normalized,cnv_calls,reports}
    mkdir -p "$LOG_DIR"
    
    print_success "Output directories created"
}

# Function to run depth extraction
run_depth_extraction() {
    print_info "Step 1: Extracting read depth per exon..."
    
    local output_dir="$RESULTS_DIR/depth"
    local log_file="$LOG_DIR/depth_extraction.log"
    
    if ! bash "$BIN_DIR/extract_depth.sh" "$INPUT_BAM_DIR" "$BED_FILE" "$output_dir" "$SAMPLE_TYPE" 2>&1 | tee "$log_file"; then
        print_error "Depth extraction failed. Check log: $log_file"
        exit 1
    fi
    
    # Check if depth files were created
    local depth_files=$(find "$output_dir" -name "*_depth.txt" | wc -l)
    if [ "$depth_files" -eq 0 ]; then
        print_error "No depth files were created"
        exit 1
    fi
    
    print_success "Depth extraction completed ($depth_files files created)"
}

# Function to calculate coverage
run_coverage_calculation() {
    print_info "Step 2: Calculating average coverage per exon..."
    
    local input_dir="$RESULTS_DIR/depth"
    local output_file="$RESULTS_DIR/depth/coverage_summary.txt"
    local log_file="$LOG_DIR/coverage_calculation.log"
    
    if ! python3 "$BIN_DIR/calculate_coverage.py" "$input_dir" "$BED_FILE" "$output_file" 2>&1 | tee "$log_file"; then
        print_error "Coverage calculation failed. Check log: $log_file"
        exit 1
    fi
    
    if [ ! -f "$output_file" ]; then
        print_error "Coverage summary file was not created"
        exit 1
    fi
    
    print_success "Coverage calculation completed"
}

# Function to perform allele counting
run_allele_counting() {
    print_info "Step 3: Performing allele-specific counting..."
    
    local output_dir="$RESULTS_DIR/allele_counts"
    local log_file="$LOG_DIR/allele_counting.log"
    
    local cmd="python3 $BIN_DIR/allele_count.py $INPUT_BAM_DIR $SNP_FILE $output_dir"
    if [ "$SAMPLE_TYPE" != "auto" ]; then
        cmd="$cmd --sample-type $SAMPLE_TYPE"
    fi
    
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        print_error "Allele counting failed. Check log: $log_file"
        exit 1
    fi
    
    local allele_file="$output_dir/allele_counts.txt"
    if [ ! -f "$allele_file" ]; then
        print_error "Allele counts file was not created"
        exit 1
    fi
    
    print_success "Allele counting completed"
}

# Function to normalize coverage
run_normalization() {
    print_info "Step 4: Normalizing coverage and calculating Z-scores..."
    
    local coverage_file="$RESULTS_DIR/depth/coverage_summary.txt"
    local sample_info_file="$RESULTS_DIR/allele_counts/sample_info.txt"
    local output_file="$RESULTS_DIR/normalized/z_scores.txt"
    local log_file="$LOG_DIR/normalization.log"
    
    if ! python3 "$BIN_DIR/normalize_coverage.py" "$coverage_file" "$sample_info_file" "$output_file" 2>&1 | tee "$log_file"; then
        print_error "Coverage normalization failed. Check log: $log_file"
        exit 1
    fi
    
    if [ ! -f "$output_file" ]; then
        print_error "Z-scores file was not created"
        exit 1
    fi
    
    print_success "Coverage normalization completed"
}

# Function to estimate copy numbers
run_copy_number_estimation() {
    print_info "Step 5: Estimating copy numbers..."
    
    local z_scores_file="$RESULTS_DIR/normalized/z_scores.txt"
    local output_file="$RESULTS_DIR/cnv_calls/copy_numbers.txt"
    local log_file="$LOG_DIR/copy_number_estimation.log"
    
    local cmd="python3 $BIN_DIR/estimate_copy_number.py $z_scores_file $output_file"
    if [ "$SKIP_PLOTS" = true ]; then
        cmd="$cmd --no-plots"
    fi
    
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        print_error "Copy number estimation failed. Check log: $log_file"
        exit 1
    fi
    
    if [ ! -f "$output_file" ]; then
        print_error "Copy numbers file was not created"
        exit 1
    fi
    
    print_success "Copy number estimation completed"
}

# Function to generate reports
run_report_generation() {
    print_info "Step 6: Generating per-sample reports..."
    
    local cn_file="$RESULTS_DIR/cnv_calls/copy_numbers.txt"
    local allele_file="$RESULTS_DIR/allele_counts/allele_counts.txt"
    local output_dir="$RESULTS_DIR/reports"
    local log_file="$LOG_DIR/report_generation.log"
    
    local cmd="python3 $BIN_DIR/generate_report.py $cn_file $allele_file $output_dir"
    if [ "$SKIP_PLOTS" = true ]; then
        cmd="$cmd --format html"
    fi
    
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        print_error "Report generation failed. Check log: $log_file"
        exit 1
    fi
    
    local report_count=$(find "$output_dir" -name "*_report.html" | wc -l)
    print_success "Report generation completed ($report_count reports created)"
}

# Function to create pipeline summary
create_summary() {
    print_info "Creating pipeline summary..."
    
    local summary_file="$RESULTS_DIR/pipeline_summary.txt"
    local sample_count=$(find "$INPUT_BAM_DIR" -name "*.bam" | wc -l)
    
    cat > "$summary_file" << EOF
SMN CNV Detection Pipeline Summary
==================================

Pipeline Run Information:
- Date: $(date)
- Pipeline Directory: $PIPELINE_DIR
- Configuration Directory: $CONFIG_DIR
- Results Directory: $RESULTS_DIR
- Input BAM Directory: $INPUT_BAM_DIR

Configuration Files:
- BED File: $BED_FILE
- SNP Configuration: $SNP_FILE

Sample Information:
- Total BAM Files: $sample_count
- Sample Type: $SAMPLE_TYPE

Output Files:
- Depth Files: $RESULTS_DIR/depth/
- Coverage Summary: $RESULTS_DIR/depth/coverage_summary.txt
- Allele Counts: $RESULTS_DIR/allele_counts/allele_counts.txt
- Z-scores: $RESULTS_DIR/normalized/z_scores.txt
- Copy Numbers: $RESULTS_DIR/cnv_calls/copy_numbers.txt
- Reports: $RESULTS_DIR/reports/

Log Files:
- All logs: $LOG_DIR/

Notes:
- Z-score thresholds: ≤-2.5 (CN=0), -2.5 to -1.5 (CN=1), -1.5 to +1.5 (CN=2), +1.5 to +2.5 (CN=3), >+2.5 (CN=4+)
- SMN1 homozygous deletion (CN=0) indicates potential SMA affected status
- SMN1 heterozygous deletion (CN=1) indicates SMA carrier status
EOF

    print_success "Pipeline summary created: $summary_file"
}

# Function to show usage
show_usage() {
    cat << EOF
SMN CNV Detection Pipeline

Usage: $0 <input_bam_dir> [OPTIONS]

REQUIRED:
    input_bam_dir       Directory containing BAM files to analyze

OPTIONS:
    --config DIR        Configuration directory (default: $CONFIG_DIR)
    --results DIR       Results directory (default: $RESULTS_DIR)
    --sample-type TYPE  Sample type: reference, test, or auto (default: auto)
    --skip-plots        Skip generating plots to speed up analysis
    --verbose           Enable verbose output
    --help              Show this help message

DESCRIPTION:
    This pipeline detects copy number variations (CNVs) in SMN1 and SMN2 genes
    from whole exome sequencing (WES) data. It processes BAM files through
    depth extraction, coverage normalization, and copy number estimation.

REQUIREMENTS:
    - samtools
    - python3 with pandas, numpy, matplotlib, seaborn, scipy

SAMPLE TYPE AUTO-DETECTION:
    When --sample-type is set to 'auto' (default), the pipeline will:
    - Classify samples with 'ref', 'control', or 'normal' in their filename as reference
    - Classify all other samples as test samples
    - You can override this by specifying --sample-type reference or --sample-type test

EXAMPLES:
    # Basic usage with auto-detection
    $0 /path/to/bam/files/
    
    # All samples are reference samples
    $0 /path/to/bam/files/ --sample-type reference
    
    # Custom output directory
    $0 /path/to/bam/files/ --results /custom/output/dir
    
    # Fast analysis without plots
    $0 /path/to/bam/files/ --skip-plots

OUTPUT:
    Results will be saved in $RESULTS_DIR with the following structure:
    - depth/: Read depth files
    - normalized/: Z-scores and reference statistics
    - cnv_calls/: Copy number estimates
    - reports/: Per-sample HTML/JSON reports
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_usage
            exit 0
            ;;
        --config)
            CONFIG_DIR="$2"
            BED_FILE="$CONFIG_DIR/smn_exons.bed"
            SNP_FILE="$CONFIG_DIR/discriminating_snps.txt"
            shift 2
            ;;
        --results)
            RESULTS_DIR="$2"
            LOG_DIR="$RESULTS_DIR/../logs"
            shift 2
            ;;
        --sample-type)
            SAMPLE_TYPE="$2"
            if [[ ! "$SAMPLE_TYPE" =~ ^(reference|test|auto)$ ]]; then
                print_error "Invalid sample type: $SAMPLE_TYPE. Must be 'reference', 'test', or 'auto'"
                exit 1
            fi
            shift 2
            ;;
        --skip-plots)
            SKIP_PLOTS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            if [ -z "$INPUT_BAM_DIR" ]; then
                INPUT_BAM_DIR="$1"
            else
                print_error "Multiple input directories specified"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if input directory was provided
if [ -z "$INPUT_BAM_DIR" ]; then
    print_error "Input BAM directory is required"
    show_usage
    exit 1
fi

# Main pipeline execution
main() {
    print_info "Starting SMN CNV Detection Pipeline"
    print_info "Input BAM directory: $INPUT_BAM_DIR"
    print_info "Configuration directory: $CONFIG_DIR"
    print_info "Results directory: $RESULTS_DIR"
    print_info "Sample type: $SAMPLE_TYPE"
    
    # Pre-flight checks
    check_dependencies
    validate_config
    setup_directories
    
    # Execute pipeline steps
    local start_time=$(date +%s)
    
    run_depth_extraction
    run_coverage_calculation
    run_allele_counting
    run_normalization
    run_copy_number_estimation
    run_report_generation
    
    # Create summary
    create_summary
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_success "Pipeline completed successfully!"
    print_info "Total runtime: ${duration} seconds"
    print_info "Results available in: $RESULTS_DIR"
    
    # Show quick results summary
    if [ -f "$RESULTS_DIR/cnv_calls/copy_numbers.txt" ]; then
        print_info "Quick Summary:"
        python3 -c "
import pandas as pd
try:
    df = pd.read_csv('$RESULTS_DIR/cnv_calls/copy_numbers.txt', sep='\t')
    samples = df['sample_id'].unique()
    print(f'  Analyzed {len(samples)} samples')
    
    # SMN1 copy number distribution
    smn1_data = df[df['exon'].str.contains('SMN1')]
    if not smn1_data.empty:
        cn_counts = smn1_data['copy_number'].value_counts().sort_index()
        print('  SMN1 copy number distribution:')
        for cn, count in cn_counts.items():
            print(f'    CN={cn}: {count} samples')
except Exception as e:
    print(f'  Could not generate quick summary: {e}')
"
    fi
    
    print_info "View individual reports in: $RESULTS_DIR/reports/"
}

# Run main function
main "$@"
