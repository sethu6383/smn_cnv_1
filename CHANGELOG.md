# Changelog

## Version 2.0 - 2025-08-06

### Major Changes
- **Removed manifest file dependency** - Pipeline now works directly with BAM directories
- **Added automatic BAM file discovery** - Scans input directory for all .bam files
- **Implemented smart sample type detection** - Auto-classifies samples based on filename patterns
- **Simplified workflow** - Easier to use with less manual configuration

### Breaking Changes
- `sample_manifest.txt` file is no longer used or required
- Command line interface changed to require input BAM directory as first argument
- Sample type now auto-detected or specified via `--sample-type` option

### New Features
- **Auto-detection of sample types**:
  - Files with 'ref', 'control', or 'normal' in name → reference samples
  - All other files → test samples
- **Flexible sample type override** with `--sample-type` option
- **Improved command line interface** with better argument parsing
- **Enhanced error checking** for BAM file discovery and validation

### Updated Components
- `extract_depth.sh`: Now accepts BAM directory instead of manifest
- `allele_count.py`: Scans directory and auto-generates sample info
- `normalize_coverage.py`: Reads sample info from generated files
- `run_pipeline.sh`: Updated to use new directory-based approach
- Documentation updated throughout

### Usage Changes

**Old Usage:**
```bash
# Edit config/sample_manifest.txt first
./run_pipeline.sh
```

**New Usage:**
```bash
# Direct directory input
./run_pipeline.sh /path/to/bam/files/

# With options
./run_pipeline.sh /path/to/bam/files/ --sample-type reference --skip-plots
```

### Migration Guide
For users upgrading from v1.0:

1. **Remove manifest file**: No longer needed
2. **Organize BAM files**: Place all BAM files in a single directory
3. **Update commands**: Use new syntax with input directory as first argument
4. **Check sample naming**: Use descriptive names for auto-detection or specify `--sample-type`

### Backward Compatibility
- Configuration files (`smn_exons.bed`, `discriminating_snps.txt`) remain unchanged
- Output format and structure remain the same
- All analysis parameters and thresholds unchanged

## Version 1.0 - Initial Release

### Features
- Manifest-based sample input
- Copy number variation detection for SMN1/SMN2
- Z-score normalization
- HTML report generation
- Support for GRCh38 coordinates
