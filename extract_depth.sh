#!/bin/bash

# extract_depth.sh - Extract read depth per exon from BAM files
# Usage: ./extract_depth.sh <input_bam_dir> <bed_file> <output_dir> [sample_type]

set -euo pipefail

# Check arguments
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 <input_bam_dir> <bed_file> <output_dir> [sample_type]"
    echo "  input_bam_dir: Directory containing BAM files"
    echo "  bed_file: Path to BED file with exon coordinates"
    echo "  output_dir: Output directory for depth files"
    echo "  sample_type: Optional sample type (reference/test, default: auto-detect)"
    exit 1
fi

INPUT_BAM_DIR="$1"
BED_FILE="$2"
OUTPUT_DIR="$3"
SAMPLE_TYPE="${4:-auto}"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Log file
LOG_FILE="${OUTPUT_DIR}/depth_extraction.log"
echo "Starting depth extraction at $(date)" > "${LOG_FILE}"
echo "Input BAM directory: $INPUT_BAM_DIR" >> "${LOG_FILE}"
echo "BED file: $BED_FILE" >> "${LOG_FILE}"
echo "Output directory: $OUTPUT_DIR" >> "${LOG_FILE}"
echo "Sample type: $SAMPLE_TYPE" >> "${LOG_FILE}"

# Check if input directory exists
if [ ! -d "$INPUT_BAM_DIR" ]; then
    echo "ERROR: Input BAM directory not found: $INPUT_BAM_DIR" | tee -a "${LOG_FILE}"
    exit 1
fi

# Find all BAM files
BAM_FILES=($(find "$INPUT_BAM_DIR" -name "*.bam" -type f))

if [ ${#BAM_FILES[@]} -eq 0 ]; then
    echo "ERROR: No BAM files found in directory: $INPUT_BAM_DIR" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "Found ${#BAM_FILES[@]} BAM files" | tee -a "${LOG_FILE}"

# Process each BAM file
for bam_path in "${BAM_FILES[@]}"; do
    # Extract sample ID from filename (remove path and .bam extension)
    sample_id=$(basename "$bam_path" .bam)
    
    echo "Processing sample: $sample_id" | tee -a "${LOG_FILE}"
    
    # Check if BAM file exists
    if [ ! -f "$bam_path" ]; then
        echo "WARNING: BAM file not found: $bam_path" | tee -a "${LOG_FILE}"
        continue
    fi
    
    # Check if BAM index exists
    if [ ! -f "${bam_path}.bai" ] && [ ! -f "${bam_path%.*}.bai" ]; then
        echo "WARNING: BAM index not found for: $bam_path" | tee -a "${LOG_FILE}"
        echo "Attempting to create index..." | tee -a "${LOG_FILE}"
        samtools index "$bam_path" 2>> "${LOG_FILE}" || {
            echo "ERROR: Failed to index BAM file: $bam_path" | tee -a "${LOG_FILE}"
            continue
        }
    fi
    
    # Output file for this sample
    depth_file="${OUTPUT_DIR}/${sample_id}_depth.txt"
    
    # Extract depth using samtools depth with BED file
    echo "Extracting depth for $sample_id..." | tee -a "${LOG_FILE}"
    samtools depth -b "$BED_FILE" -a "$bam_path" > "$depth_file" 2>> "${LOG_FILE}"
    
    # Check if depth extraction was successful
    if [ $? -eq 0 ] && [ -s "$depth_file" ]; then
        echo "Successfully extracted depth for $sample_id" | tee -a "${LOG_FILE}"
        
        # Create sample info file
        info_file="${OUTPUT_DIR}/${sample_id}_info.txt"
        echo -e "sample_id\tbam_path\tsample_type" > "$info_file"
        
        # Auto-detect sample type if not specified
        if [ "$SAMPLE_TYPE" = "auto" ]; then
            # Simple heuristic: samples with "ref" or "control" in name are reference
            if [[ "$sample_id" =~ [Rr]ef|[Cc]ontrol|[Nn]ormal ]]; then
                detected_type="reference"
            else
                detected_type="test"
            fi
        else
            detected_type="$SAMPLE_TYPE"
        fi
        
        echo -e "${sample_id}\t${bam_path}\t${detected_type}" >> "$info_file"
        echo "Sample type assigned: $detected_type" | tee -a "${LOG_FILE}"
    else
        echo "ERROR: Depth extraction failed for $sample_id" | tee -a "${LOG_FILE}"
    fi
    
done

echo "Depth extraction completed at $(date)" | tee -a "${LOG_FILE}"
echo "Results saved in: $OUTPUT_DIR"

# Create summary
echo "Creating extraction summary..." | tee -a "${LOG_FILE}"
successful_samples=$(find "$OUTPUT_DIR" -name "*_depth.txt" | wc -l)
echo "Successfully processed $successful_samples out of ${#BAM_FILES[@]} BAM files" | tee -a "${LOG_FILE}"
