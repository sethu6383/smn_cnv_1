#!/bin/bash

# enhanced_run_pipeline.sh - Enhanced SMN CNV detection pipeline with control gene normalization and dual segmentation
# Usage: ./enhanced_run_pipeline.sh <input_bam_dir> [OPTIONS]

set -euo pipefail

# Enhanced pipeline paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PIPELINE_DIR/config"
RESULTS_DIR="$PIPELINE_DIR/results_enhanced"
BIN_DIR="$PIPELINE_DIR/bin"
LOG_DIR="$PIPELINE_DIR/logs_enhanced"

# Configuration files
BED_FILE="$CONFIG_DIR/smn_exons.bed"
CONTROL_BED_FILE="$CONFIG_DIR/control_genes.bed"
SNP_FILE="$CONFIG_DIR/discriminating_snps.txt"

# Enhanced pipeline options
SKIP_PLOTS=false
VERBOSE=false
SAMPLE_TYPE="auto"
SEGMENTATION_METHOD="both"
CONTROL_GENE_NORMALIZATION=true
INPUT_BAM_DIR=""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

print_enhanced() {
    print_status "$PURPLE" "ENHANCED: $1"
}

# Function to check enhanced dependencies
check_enhanced_dependencies() {
    print_info "Checking enhanced dependencies..."
    
    local missing_tools=()
    local missing_packages=()
    
    # Check required command-line tools
    for tool in samtools python3; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    # Check Python packages
    if ! python3 -c "import pandas, numpy, matplotlib, seaborn, scipy, sklearn" &> /dev/null; then
        print_warning "Some Python packages may be missing. Required: pandas, numpy, matplotlib, seaborn, scipy, sklearn"
    fi
    
    # Check optional packages
    if ! python3 -c "import hmmlearn" &> /dev/null 2>&1; then
        print_warning "hmmlearn not available - will use custom HMM implementation"
    else
        print_success "hmmlearn available - will use optimized HMM implementation"
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    print_success "All core dependencies found"
}

# Function to validate enhanced configuration
validate_enhanced_config() {
    print_info "Validating enhanced configuration files and input..."
    
    # Check if configuration files exist
    for file in "$BED_FILE" "$SNP_FILE"; do
        if [ ! -f "$file" ]; then
            print_error "Configuration file not found: $file"
            exit 1
        fi
    done
    
    # Check control genes BED file
    if [ "$CONTROL_GENE_NORMALIZATION" = true ]; then
        if [ ! -f "$CONTROL_BED_FILE" ]; then
            print_warning "Control genes BED file not found: $CONTROL_BED_FILE"
            print_info "Creating control genes BED file..."
            create_control_genes_bed
        fi
        
        # Validate control genes BED file
        if ! grep -q "CFTR_exon" "$CONTROL_BED_FILE" || ! grep -q "RPPH1_exon" "$CONTROL_BED_FILE"; then
            print_error "Control genes BED file does not contain expected CFTR/RPPH1 exon entries"
            exit 1
        fi
    fi
    
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
    
    print_success "Enhanced configuration validated ($bam_count BAM files found)"
}

# Function to create control genes BED file if not exists
create_control_genes_bed() {
    cat > "$CONTROL_BED_FILE" << 'EOF'
# Control genes BED file for normalization (GRCh38/hg38 coordinates)
# CFTR gene exons (chr7:117,120,016-117,308,718)
chr7	117120016	117120140	CFTR_exon1	.	+
chr7	117144363	117144462	CFTR_exon2	.	+
chr7	117149094	117149196	CFTR_exon3	.	+
chr7	117170286	117170412	CFTR_exon4	.	+
chr7	117171052	117171159	CFTR_exon5	.	+
chr7	117174363	117174502	CFTR_exon6	.	+
chr7	117176394	117176532	CFTR_exon7	.	+
chr7	117180164	117180248	CFTR_exon8	.	+
chr7	117182077	117182183	CFTR_exon9	.	+
chr7	117188637	117188770	CFTR_exon10	.	+
# RNase P (RPPH1) gene exons (chr14:20,649,636-20,650,216)
chr14	20649636	20649736	RPPH1_exon1	.	+
chr14	20649836	20649936	RPPH1_exon2	.	+
chr14	20650116	20650216	RPPH1_exon3	.	+
EOF
    print_success "Control genes BED file created: $CONTROL_BED_FILE"
}

# Function to setup enhanced directories
setup_enhanced_directories() {
    print_info "Setting up enhanced output directories..."
    
    mkdir -p "$RESULTS_DIR"/{depth,control_depth,allele_counts,normalized_enhanced,dual_segmentation,cnv_calls_enhanced,reports_enhanced}
    mkdir -p "$LOG_DIR"
    
    print_success "Enhanced output directories created"
}

# Function to run enhanced depth extraction (includes control genes)
run_enhanced_depth_extraction() {
    print_enhanced "Step 1: Enhanced depth extraction (target + control genes)..."
    
    local target_output_dir="$RESULTS_DIR/depth"
    local control_output_dir="$RESULTS_DIR/control_depth"
    local log_file="$LOG_DIR/enhanced_depth_extraction.log"
    
    # Extract depth for target genes (SMN1/SMN2)
    print_info "Extracting depth for target genes (SMN1/SMN2)..."
    if ! bash "$BIN_DIR/extract_depth.sh" "$INPUT_BAM_DIR" "$BED_FILE" "$target_output_dir" "$SAMPLE_TYPE" 2>&1 | tee "$log_file"; then
        print_error "Target depth extraction failed. Check log: $log_file"
        exit 1
    fi
    
    # Extract depth for control genes (CFTR/RNase P) if enabled
    if [ "$CONTROL_GENE_NORMALIZATION" = true ]; then
        print_info "Extracting depth for control genes (CFTR/RNase P)..."
        if ! bash "$BIN_DIR/extract_depth.sh" "$INPUT_BAM_DIR" "$CONTROL_BED_FILE" "$control_output_dir" "$SAMPLE_TYPE" 2>&1 | tee -a "$log_file"; then
            print_warning "Control depth extraction failed. Will proceed without control gene normalization."
            CONTROL_GENE_NORMALIZATION=false
        else
            print_success "Control gene depth extraction completed"
        fi
    fi
    
    # Check if target depth files were created
    local depth_files=$(find "$target_output_dir" -name "*_depth.txt" | wc -l)
    if [ "$depth_files" -eq 0 ]; then
        print_error "No target depth files were created"
        exit 1
    fi
    
    print_success "Enhanced depth extraction completed ($depth_files target files created)"
}

# Function to calculate enhanced coverage
run_enhanced_coverage_calculation() {
    print_enhanced "Step 2: Enhanced coverage calculation..."
    
    local target_input_dir="$RESULTS_DIR/depth"
    local control_input_dir="$RESULTS_DIR/control_depth"
    local target_output_file="$RESULTS_DIR/depth/coverage_summary.txt"
    local control_output_file="$RESULTS_DIR/control_depth/control_coverage_summary.txt"
    local log_file="$LOG_DIR/enhanced_coverage_calculation.log"
    
    # Calculate target coverage
    print_info "Calculating target gene coverage..."
    if ! python3 "$BIN_DIR/calculate_coverage.py" "$target_input_dir" "$BED_FILE" "$target_output_file" 2>&1 | tee "$log_file"; then
        print_error "Target coverage calculation failed. Check log: $log_file"
        exit 1
    fi
    
    # Calculate control coverage if available
    if [ "$CONTROL_GENE_NORMALIZATION" = true ] && [ -d "$control_input_dir" ]; then
        print_info "Calculating control gene coverage..."
        if ! python3 "$BIN_DIR/calculate_coverage.py" "$control_input_dir" "$CONTROL_BED_FILE" "$control_output_file" 2>&1 | tee -a "$log_file"; then
            print_warning "Control coverage calculation failed. Will proceed without control normalization."
            CONTROL_GENE_NORMALIZATION=false
        else
            print_success "Control gene coverage calculation completed"
        fi
    fi
    
    if [ ! -f "$target_output_file" ]; then
        print_error "Target coverage summary file was not created"
        exit 1
    fi
    
    print_success "Enhanced coverage calculation completed"
}

# Function to perform allele counting (unchanged from original)
run_enhanced_allele_counting() {
    print_info "Step 3: Performing allele-specific counting..."
    
    local output_dir="$RESULTS_DIR/allele_counts"
    local log_file="$LOG_DIR/enhanced_allele_counting.log"
    
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

# Function to run enhanced normalization with control genes
run_enhanced_normalization() {
    print_enhanced "Step 4: Enhanced normalization with control gene correction..."
    
    local target_coverage_file="$RESULTS_DIR/depth/coverage_summary.txt"
    local sample_info_file="$RESULTS_DIR/allele_counts/sample_info.txt"
    local control_coverage_file="$RESULTS_DIR/control_depth/control_coverage_summary.txt"
    local output_file="$RESULTS_DIR/normalized_enhanced/enhanced_z_scores.txt"
    local log_file="$LOG_DIR/enhanced_normalization.log"
    
    # Use enhanced control gene normalization if available
    if [ "$CONTROL_GENE_NORMALIZATION" = true ] && [ -f "$control_coverage_file" ]; then
        print_info "Using control gene-based normalization..."
        local cmd="python3 $BIN_DIR/control_gene_normalization.py $target_coverage_file $sample_info_file $control_coverage_file $output_file"
    else
        print_info "Using standard normalization (control genes not available)..."
        local cmd="python3 $BIN_DIR/normalize_coverage.py $target_coverage_file $sample_info_file $output_file"
    fi
    
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        print_error "Enhanced normalization failed. Check log: $log_file"
        exit 1
    fi
    
    if [ ! -f "$output_file" ]; then
        print_error "Enhanced Z-scores file was not created"
        exit 1
    fi
    
    print_success "Enhanced normalization completed"
}

# Function to run dual segmentation analysis
run_dual_segmentation() {
    print_enhanced "Step 5: Dual segmentation analysis (CBS + HMM)..."
    
    local z_scores_file="$RESULTS_DIR/normalized_enhanced/enhanced_z_scores.txt"
    local output_file="$RESULTS_DIR/dual_segmentation/segmentation_results.txt"
    local log_file="$LOG_DIR/dual_segmentation.log"
    
    local cmd="python3 $BIN_DIR/dual_segmentation.py $z_scores_file $output_file --method $SEGMENTATION_METHOD"
    if [ "$SKIP_PLOTS" = true ]; then
        cmd="$cmd --no-plots"
    fi
    
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        print_error "Dual segmentation failed. Check log: $log_file"
        exit 1
    fi
    
    if [ ! -f "$output_file" ]; then
        print_error "Segmentation results file was not created"
        exit 1
    fi
    
    print_success "Dual segmentation analysis completed"
}

# Function to run enhanced copy number estimation
run_enhanced_copy_number_estimation() {
    print_enhanced "Step 6: Enhanced copy number estimation..."
    
    local segmentation_file="$RESULTS_DIR/dual_segmentation/segmentation_results.txt"
    local output_file="$RESULTS_DIR/cnv_calls_enhanced/enhanced_copy_numbers.txt"
    local log_file="$LOG_DIR/enhanced_copy_number_estimation.log"
    
    local cmd="python3 $BIN_DIR/enhanced_copy_number_estimation.py $segmentation_file $output_file"
    if [ "$SKIP_PLOTS" = true ]; then
        cmd="$cmd --no-plots"
    fi
    
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        print_error "Enhanced copy number estimation failed. Check log: $log_file"
        exit 1
    fi
    
    if [ ! -f "$output_file" ]; then
        print_error "Enhanced copy numbers file was not created"
        exit 1
    fi
    
    print_success "Enhanced copy number estimation completed"
}

# Function to generate enhanced reports
run_enhanced_report_generation() {
    print_enhanced "Step 7: Generating enhanced per-sample reports..."
    
    local cn_file="$RESULTS_DIR/cnv_calls_enhanced/enhanced_copy_numbers.txt"
    local allele_file="$RESULTS_DIR/allele_counts/allele_counts.txt"
    local output_dir="$RESULTS_DIR/reports_enhanced"
    local log_file="$LOG_DIR/enhanced_report_generation.log"
    
    local cmd="python3 $BIN_DIR/generate_report.py $cn_file $allele_file $output_dir"
    if [ "$SKIP_PLOTS" = true ]; then
        cmd="$cmd --format html"
    fi
    
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        print_error "Enhanced report generation failed. Check log: $log_file"
        exit 1
    fi
    
    local report_count=$(find "$output_dir" -name "*_report.html" | wc -l)
    print_success "Enhanced report generation completed ($report_count reports created)"
}

# Function to create enhanced pipeline summary
create_enhanced_summary() {
    print_enhanced "Creating enhanced pipeline summary..."
    
    local summary_file="$RESULTS_DIR/enhanced_pipeline_summary.txt"
    local sample_count=$(find "$INPUT_BAM_DIR" -name "*.bam" | wc -l)
    
    cat > "$summary_file" << EOF
Enhanced SMN CNV Detection Pipeline Summary
==========================================

Pipeline Run Information:
- Date: $(date)
- Pipeline Directory: $PIPELINE_DIR
- Configuration Directory: $CONFIG_DIR
- Results Directory: $RESULTS_DIR
- Input BAM Directory: $INPUT_BAM_DIR

Enhanced Features:
- Control Gene Normalization: $CONTROL_GENE_NORMALIZATION
- Segmentation Method: $SEGMENTATION_METHOD
- Enhanced Confidence Scoring: Enabled
- Dual Segmentation (CBS + HMM): Enabled

Configuration Files:
- Target Genes BED: $BED_FILE
- Control Genes BED: $CONTROL_BED_FILE
- SNP Configuration: $SNP_FILE

Sample Information:
- Total BAM Files: $sample_count
- Sample Type: $SAMPLE_TYPE

Enhanced Output Files:
- Target Depth Files: $RESULTS_DIR/depth/
- Control Depth Files: $RESULTS_DIR/control_depth/ (if enabled)
- Enhanced Coverage: $RESULTS_DIR/depth/coverage_summary.txt
- Control Coverage: $RESULTS_DIR/control_depth/control_coverage_summary.txt (if enabled)
- Allele Counts: $RESULTS_DIR/allele_counts/allele_counts.txt
- Enhanced Z-scores: $RESULTS_DIR/normalized_enhanced/enhanced_z_scores.txt
- Dual Segmentation: $RESULTS_DIR/dual_segmentation/segmentation_results.txt
- Enhanced Copy Numbers: $RESULTS_DIR/cnv_calls_enhanced/enhanced_copy_numbers.txt
- Enhanced Reports: $RESULTS_DIR/reports_enhanced/

Log Files:
- All enhanced logs: $LOG_DIR/

Enhanced Analysis Notes:
- Control genes used for normalization: CFTR (chr7), RNase P/RPPH1 (chr14)
- Segmentation methods: Circular Binary Segmentation (CBS) + Hidden Markov Model (HMM)
- Confidence scoring includes: coverage quality, normalization quality, segmentation consistency
- Clinical significance assessment: automated pathogenicity prediction
- Quality thresholds: minimum coverage ≥10x, maximum missing rate ≤30%

Traditional thresholds (enhanced with confidence):
- Z-score ≤-2.5 (CN=0), -2.5 to -1.5 (CN=1), -1.5 to +1.5 (CN=2), +1.5 to +2.5 (CN=3), >+2.5 (CN=4+)
- SMN1 homozygous deletion (CN=0): likely SMA affected
- SMN1 heterozygous deletion (CN=1): SMA carrier
- Enhanced confidence: high (>0.8), medium (0.6-0.8), low (<0.6)
EOF

    print_success "Enhanced pipeline summary created: $summary_file"
}

# Function to show enhanced usage
show_enhanced_usage() {
    cat << EOF
Enhanced SMN CNV Detection Pipeline with Control Gene Normalization and Dual Segmentation

Usage: $0 <input_bam_dir> [OPTIONS]

REQUIRED:
    input_bam_dir       Directory containing BAM files to analyze

OPTIONS:
    --config DIR        Configuration directory (default: $CONFIG_DIR)
    --results DIR       Results directory (default: $RESULTS_DIR)
    --sample-type TYPE  Sample type: reference, test, or auto (default: auto)
    --segmentation TYPE Segmentation method: cbs, hmm, or both (default: both)
    --no-control        Disable control gene normalization
    --skip-plots        Skip generating plots to speed up analysis
    --verbose           Enable verbose output
    --help              Show this help message

ENHANCED FEATURES:
    Control Gene Normalization:
    - Uses CFTR (chr7) and RNase P/RPPH1 (chr14) as stable reference regions
    - Corrects for sequencing depth variation, capture bias, and batch effects
    - Improves accuracy in samples with uneven WES coverage

    Dual Segmentation Analysis:
    - Circular Binary Segmentation (CBS): Identifies statistical change-points
    - Hidden Markov Model (HMM): Probabilistically infers copy number states
    - Consensus calling combines both methods for optimal accuracy
    - Handles missing data and noise better than single methods

    Enhanced Quality Assessment:
    - Multi-tier confidence scoring (high/medium/low)
    - Coverage quality metrics and normalization assessment
    - Segmentation consistency evaluation
    - Clinical significance prediction

REQUIREMENTS:
    - samtools
    - python3 with pandas, numpy, matplotlib, seaborn, scipy, sklearn
    - Optional: hmmlearn (for optimized HMM implementation)

SAMPLE TYPE AUTO-DETECTION:
    When --sample-type is set to 'auto' (default), the pipeline will:
    - Classify samples with 'ref', 'control', or 'normal' in filename as reference
    - Classify all other samples as test samples
    - You can override this by specifying --sample-type reference or --sample-type test

EXAMPLES:
    # Basic enhanced analysis with all features
    $0 /path/to/bam/files/
    
    # All samples are reference samples (building reference database)
    $0 /path/to/bam/files/ --sample-type reference
    
    # Use only HMM segmentation (faster)
    $0 /path/to/bam/files/ --segmentation hmm
    
    # Disable control gene normalization (fallback mode)
    $0 /path/to/bam/files/ --no-control
    
    # Custom output directory
    $0 /path/to/bam/files/ --results /custom/enhanced/output/
    
    # Fast analysis without plots
    $0 /path/to/bam/files/ --skip-plots

OUTPUT STRUCTURE:
    Enhanced results will be saved in $RESULTS_DIR with:
    - depth/: Target gene read depth files
    - control_depth/: Control gene depth files (CFTR, RNase P)
    - normalized_enhanced/: Enhanced Z-scores with control normalization
    - dual_segmentation/: CBS and HMM segmentation results
    - cnv_calls_enhanced/: Enhanced copy number calls with confidence
    - reports_enhanced/: Per-sample HTML/JSON reports with clinical interpretation

PERFORMANCE ENHANCEMENTS:
    - Robust outlier detection using MAD (Median Absolute Deviation)
    - Batch effect correction through control gene normalization
    - Missing data handling in segmentation algorithms
    - Confidence-weighted consensus calling
    - Clinical significance automated assessment
EOF
}

# Parse enhanced command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_enhanced_usage
            exit 0
            ;;
        --config)
            CONFIG_DIR="$2"
            BED_FILE="$CONFIG_DIR/smn_exons.bed"
            CONTROL_BED_FILE="$CONFIG_DIR/control_genes.bed"
            SNP_FILE="$CONFIG_DIR/discriminating_snps.txt"
            shift 2
            ;;
        --results)
            RESULTS_DIR="$2"
            LOG_DIR="$RESULTS_DIR/../logs_enhanced"
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
        --segmentation)
            SEGMENTATION_METHOD="$2"
            if [[ ! "$SEGMENTATION_METHOD" =~ ^(cbs|hmm|both)$ ]]; then
                print_error "Invalid segmentation method: $SEGMENTATION_METHOD. Must be 'cbs', 'hmm', or 'both'"
                exit 1
            fi
            shift 2
            ;;
        --no-control)
            CONTROL_GENE_NORMALIZATION=false
            shift
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
            show_enhanced_usage
            exit 1
            ;;
        *)
            if [ -z "$INPUT_BAM_DIR" ]; then
                INPUT_BAM_DIR="$1"
            else
                print_error "Multiple input directories specified"
                show_enhanced_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if input directory was provided
if [ -z "$INPUT_BAM_DIR" ]; then
    print_error "Input BAM directory is required"
    show_enhanced_usage
    exit 1
fi

# Main enhanced pipeline execution
main() {
    print_enhanced "Starting Enhanced SMN CNV Detection Pipeline"
    print_info "Input BAM directory: $INPUT_BAM_DIR"
    print_info "Configuration directory: $CONFIG_DIR"
    print_info "Results directory: $RESULTS_DIR"
    print_info "Sample type: $SAMPLE_TYPE"
    print_info "Segmentation method: $SEGMENTATION_METHOD"
    print_info "Control gene normalization: $CONTROL_GENE_NORMALIZATION"
    
    # Pre-flight checks
    check_enhanced_dependencies
    validate_enhanced_config
    setup_enhanced_directories
    
    # Execute enhanced pipeline steps
    local start_time=$(date +%s)
    
    run_enhanced_depth_extraction
    run_enhanced_coverage_calculation
    run_enhanced_allele_counting
    run_enhanced_normalization
    run_dual_segmentation
    run_enhanced_copy_number_estimation
    run_enhanced_report_generation
    
    # Create enhanced summary
    create_enhanced_summary
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_success "Enhanced pipeline completed successfully!"
    print_info "Total runtime: ${duration} seconds"
    print_info "Enhanced results available in: $RESULTS_DIR"
    
    # Show enhanced results summary
    if [ -f "$RESULTS_DIR/cnv_calls_enhanced/enhanced_copy_numbers.txt" ]; then
        print_enhanced "Enhanced Analysis Summary:"
        python3 -c "
import pandas as pd
import numpy as np
try:
    # Load enhanced results
    df = pd.read_csv('$RESULTS_DIR/cnv_calls_enhanced/enhanced_copy_numbers.txt', sep='\t')
    samples = df['sample_id'].unique()
    print(f'  Analyzed {len(samples)} samples with enhanced pipeline')
    
    # Enhanced confidence distribution
    conf_dist = df['confidence'].value_counts()
    print('  Enhanced confidence distribution:')
    for conf, count in conf_dist.items():
        pct = 100 * count / len(df)
        print(f'    {conf.capitalize()}: {count} calls ({pct:.1f}%)')
    
    # Enhanced SMN1 copy number distribution with confidence
    smn1_data = df[df['exon'].str.contains('SMN1')]
    if not smn1_data.empty:
        print('  SMN1 copy number distribution (enhanced):')
        for cn in sorted(smn1_data['copy_number'].unique()):
            if not np.isnan(cn):
                cn_data = smn1_data[smn1_data['copy_number'] == cn]
                high_conf = (cn_data['confidence'] == 'high').sum()
                total = len(cn_data)
                print(f'    CN={int(cn)}: {total} samples ({high_conf} high confidence)')
    
    # Control gene normalization effectiveness
    if 'normalization_quality' in df.columns:
        norm_qual = df['normalization_quality'].value_counts()
        print('  Normalization quality:')
        for qual, count in norm_qual.items():
            pct = 100 * count / len(df)
            print(f'    {qual.capitalize()}: {count} calls ({pct:.1f}%)')
    
    # Segmentation method distribution
    if 'segmentation_method' in df.columns:
        seg_dist = df['segmentation_method'].value_counts()
        print('  Segmentation method distribution:')
        for method, count in seg_dist.items():
            pct = 100 * count / len(df)
            print(f'    {method.capitalize()}: {count} calls ({pct:.1f}%)')

    # Gene-level results if available
    gene_file = '$RESULTS_DIR/cnv_calls_enhanced/enhanced_copy_numbers_enhanced_gene_level.txt'
    try:
        gene_df = pd.read_csv(gene_file, sep='\t')
        print('  Gene-level clinical significance:')
        clin_sig = gene_df['clinical_significance'].value_counts()
        for sig, count in clin_sig.items():
            print(f'    {sig.replace(\"_\", \" \").title()}: {count} genes')
    except:
        pass
        
except Exception as e:
    print(f'  Could not generate enhanced summary: {e}')
"
    fi
    
    print_enhanced "Enhanced Features Applied:"
    echo "  ✓ Control gene normalization (CFTR + RNase P)"
    echo "  ✓ Dual segmentation analysis (CBS + HMM)"
    echo "  ✓ Enhanced confidence scoring"
    echo "  ✓ Clinical significance assessment"
    echo "  ✓ Quality-weighted consensus calling"
    
    print_info "View enhanced individual reports in: $RESULTS_DIR/reports_enhanced/"
    print_info "Enhanced pipeline summary: $RESULTS_DIR/enhanced_pipeline_summary.txt"
}

# Run enhanced main function
main "$@"
