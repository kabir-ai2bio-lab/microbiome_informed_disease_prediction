#!/usr/bin/env python3
"""
Shotgun Sequencing Data Preprocessing Pipeline for MVIB

This pipeline processes raw shotgun metagenomic sequencing data (FASTQ files)
into the format required by the MVIB model:
- Species-level relative abundance profiles (from MetaPhlAn2)
- Strain-level marker gene profiles

Pipeline Steps:
1. Quality Control (FastQC)
2. Read Trimming/Filtering (Trimmomatic)
3. Species Profiling (MetaPhlAn2)
4. Marker Gene Extraction (custom processing)
5. Format Conversion (to MVIB format)

Author: Generated for MVIB preprocessing
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import json

# Default paths for this project (real raw data location)
MVIB_ROOT = Path(__file__).parent.parent
DEFAULT_FASTQ_DIR = MVIB_ROOT / 'fastq_downloads'
DEFAULT_OUTPUT_DIR = MVIB_ROOT / 'processed_data'
DEFAULT_SAMPLE_SHEET = MVIB_ROOT / 'fastq_downloads' / 'sample_sheet.csv'

# Ensure venv bin is in PATH when running from venv (finds pip-installed tools)
_venv_bin = Path(sys.executable).resolve().parent
if (_venv_bin / 'fastqc').exists() or (_venv_bin / 'metaphlan').exists():
    os.environ['PATH'] = str(_venv_bin) + os.pathsep + os.environ.get('PATH', '')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ShotgunPreprocessor:
    """
    Main preprocessing pipeline for shotgun metagenomic sequencing data.
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        sample_sheet: str,
        metaphlan_db: Optional[str] = None,
        threads: int = 8,
        min_read_length: int = 50,
        quality_threshold: int = 20
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            input_dir: Directory containing raw FASTQ files
            output_dir: Directory for processed outputs
            sample_sheet: CSV file with sample metadata (sampleID, disease, etc.)
            metaphlan_db: Path to MetaPhlAn2 database (optional, uses default if None)
            threads: Number of threads for parallel processing
            min_read_length: Minimum read length after trimming
            quality_threshold: Quality score threshold for trimming
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sample_sheet = Path(sample_sheet)
        self.metaphlan_db = metaphlan_db
        self.threads = threads
        self.min_read_length = min_read_length
        self.quality_threshold = quality_threshold
        self.metaphlan_cmd = None  # Set by check_dependencies: 'metaphlan' or 'metaphlan2.py'
        self.use_fasttrimmatic = False  # Set by check_dependencies if fasttrimmatic found
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'qc').mkdir(exist_ok=True)
        (self.output_dir / 'trimmed').mkdir(exist_ok=True)
        (self.output_dir / 'metaphlan').mkdir(exist_ok=True)
        (self.output_dir / 'markers').mkdir(exist_ok=True)
        (self.output_dir / 'formatted').mkdir(exist_ok=True)
        
        # Load sample sheet
        self.samples_df = pd.read_csv(self.sample_sheet)
        logger.info(f"Loaded {len(self.samples_df)} samples from sample sheet")
        
    def check_dependencies(self) -> bool:
        """
        Check if required tools are installed.
        Supports both MetaPhlAn3/4 (metaphlan) and MetaPhlAn2 (metaphlan2.py).
        
        Returns:
            True if all dependencies are available
        """
        # FastQC (pip: fastqc-py)
        self.use_fasttrimmatic = False
        try:
            result = subprocess.run(['which', 'fastqc'], capture_output=True, text=True)
            if result.returncode != 0:
                missing = ['FastQC']
            else:
                missing = []
        except Exception:
            missing = ['FastQC']
        
        # Trimmomatic or fasttrimmatic (pip: fasttrimmatic)
        self.use_fasttrimmatic = False
        try:
            trimmomatic = subprocess.run(['which', 'trimmomatic'], capture_output=True, text=True)
            if trimmomatic.returncode != 0:
                fasttrimmatic = subprocess.run(['which', 'fasttrimmatic'], capture_output=True, text=True)
                if fasttrimmatic.returncode == 0:
                    self.use_fasttrimmatic = True
                    logger.info("Using fasttrimmatic (Trimmomatic alternative)")
                else:
                    missing.append('Trimmomatic (or fasttrimmatic)')
        except Exception:
            missing.append('Trimmomatic')
        
        # Bowtie2 (conda only on Linux - no pip package)
        try:
            result = subprocess.run(['which', 'bowtie2'], capture_output=True, text=True)
            if result.returncode != 0:
                missing.append('Bowtie2')
        except Exception:
            missing.append('Bowtie2')
        
        # MetaPhlAn: prefer 'metaphlan' (MetaPhlAn4) - has working DB. metaphlan2.py DB download often fails (Bitbucket deprecated)
        metaphlan_found = False
        for cmd in ['metaphlan', 'metaphlan2.py']:
            try:
                result = subprocess.run(
                    ['which', cmd],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    self.metaphlan_cmd = cmd
                    metaphlan_found = True
                    logger.info(f"Using MetaPhlAn: {cmd}")
                    break
            except Exception:
                pass
        if not metaphlan_found:
            missing.append('MetaPhlAn (metaphlan or metaphlan2.py)')
        
        if missing:
            logger.error(f"Missing required tools: {', '.join(missing)}")
            logger.error("Please install: conda install -c bioconda fastqc trimmomatic metaphlan bowtie2")
            return False
        
        logger.info("All required tools are available")
        return True
    
    def run_quality_control(self, sample_id: str, fastq_files: List[str]) -> bool:
        """
        Run FastQC quality control on FASTQ files.
        
        Args:
            sample_id: Sample identifier
            fastq_files: List of FASTQ file paths
            
        Returns:
            True if successful
        """
        logger.info(f"Running quality control for {sample_id}")
        
        qc_output = self.output_dir / 'qc' / sample_id
        qc_output.mkdir(exist_ok=True)
        
        try:
            cmd = ['fastqc', '-o', str(qc_output), '-t', str(self.threads)] + fastq_files
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Quality control completed for {sample_id}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FastQC failed for {sample_id}: {e.stderr}")
            return False
    
    def trim_reads(
        self,
        sample_id: str,
        forward_reads: str,
        reverse_reads: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Trim reads using Trimmomatic.
        
        Args:
            sample_id: Sample identifier
            forward_reads: Path to forward reads (R1)
            reverse_reads: Path to reverse reads (R2), optional for single-end
            
        Returns:
            Tuple of (trimmed_forward, trimmed_reverse) paths
        """
        logger.info(f"Trimming reads for {sample_id}")
        
        trimmed_dir = self.output_dir / 'trimmed'
        forward_trimmed = trimmed_dir / f"{sample_id}_R1_trimmed.fastq.gz"
        reverse_trimmed = trimmed_dir / f"{sample_id}_R2_trimmed.fastq.gz" if reverse_reads else None
        
        if self.use_fasttrimmatic:
            # fasttrimmatic (pip install fasttrimmatic)
            if reverse_reads:
                cmd = [
                    'fasttrimmatic', '--paired',
                    '-i1', str(forward_reads), '-i2', str(reverse_reads),
                    '-o1', str(forward_trimmed), '-o2', str(reverse_trimmed),
                    '--leading-q', str(self.quality_threshold),
                    '--trailing-q', str(self.quality_threshold),
                    '--window', '4', '--min-avg', str(self.quality_threshold),
                    '--min-len', str(self.min_read_length),
                    '--max-workers', str(self.threads)
                ]
            else:
                cmd = [
                    'fasttrimmatic', '-i', str(forward_reads), '-o', str(forward_trimmed),
                    '--leading-q', str(self.quality_threshold),
                    '--trailing-q', str(self.quality_threshold),
                    '--window', '4', '--min-avg', str(self.quality_threshold),
                    '--min-len', str(self.min_read_length),
                    '--max-workers', str(self.threads)
                ]
        else:
            # Trimmomatic (conda install trimmomatic)
            if reverse_reads:
                cmd = [
                    'trimmomatic', 'PE',
                    '-threads', str(self.threads),
                    '-phred33',
                    str(forward_reads), str(reverse_reads),
                    str(forward_trimmed), str(trimmed_dir / f"{sample_id}_R1_unpaired.fastq.gz"),
                    str(reverse_trimmed), str(trimmed_dir / f"{sample_id}_R2_unpaired.fastq.gz"),
                    f'ILLUMINACLIP:TruSeq3-PE.fa:2:30:10',
                    f'LEADING:{self.quality_threshold}',
                    f'TRAILING:{self.quality_threshold}',
                    f'SLIDINGWINDOW:4:{self.quality_threshold}',
                    f'MINLEN:{self.min_read_length}'
                ]
            else:
                cmd = [
                    'trimmomatic', 'SE',
                    '-threads', str(self.threads),
                    '-phred33',
                    str(forward_reads),
                    str(forward_trimmed),
                    f'ILLUMINACLIP:TruSeq3-SE.fa:2:30:10',
                    f'LEADING:{self.quality_threshold}',
                    f'TRAILING:{self.quality_threshold}',
                    f'SLIDINGWINDOW:4:{self.quality_threshold}',
                    f'MINLEN:{self.min_read_length}'
                ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Trimming completed for {sample_id}")
            return str(forward_trimmed), str(reverse_trimmed) if reverse_trimmed else None
        except subprocess.CalledProcessError as e:
            logger.error(f"Trimming failed for {sample_id}: {e.stderr}")
            raise
    
    def run_metaphlan2(
        self,
        sample_id: str,
        forward_reads: str,
        reverse_reads: Optional[str] = None
    ) -> str:
        """
        Run MetaPhlAn2 to generate species abundance profiles.
        
        Args:
            sample_id: Sample identifier
            forward_reads: Path to trimmed forward reads
            reverse_reads: Path to trimmed reverse reads (optional)
            
        Returns:
            Path to MetaPhlAn2 output file
        """
        logger.info(f"Running MetaPhlAn for {sample_id} (using {self.metaphlan_cmd})")
        
        metaphlan_output = self.output_dir / 'metaphlan' / f"{sample_id}_metaphlan.txt"
        mapout_file = self.output_dir / 'metaphlan' / f"{sample_id}.bowtie2.bz2"
        
        # MetaPhlAn 4 uses --mapout (not --bowtie2out) and -1/-2 for paired-end
        if self.metaphlan_cmd == 'metaphlan2.py':
            # MetaPhlAn2: comma-separated input, --bowtie2out
            input_reads = str(forward_reads)
            if reverse_reads:
                input_reads = f"{forward_reads},{reverse_reads}"
            cmd = [
                self.metaphlan_cmd,
                input_reads,
                '--input_type', 'fastq',
                '--nproc', str(self.threads),
                '--bowtie2out', str(mapout_file),
                '-o', str(metaphlan_output),
                '--tax_lev', 's',
                '--min_cu_len', '2000',
                '--stat_q', '0.1'
            ]
            if self.metaphlan_db:
                cmd.extend(['--bowtie2db', self.metaphlan_db])
        else:
            # MetaPhlAn 3/4: --mapout, -1/-2 for paired-end
            cmd = [
                self.metaphlan_cmd,
                '--input_type', 'fastq',
                '--nproc', str(self.threads),
                '--mapout', str(mapout_file),
                '-o', str(metaphlan_output)
            ]
            if self.metaphlan_db:
                cmd.extend(['--db_dir', self.metaphlan_db])
            if reverse_reads:
                cmd.extend(['-1', str(forward_reads), '-2', str(reverse_reads)])
            else:
                cmd.append(str(forward_reads))
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"MetaPhlAn2 completed for {sample_id}")
            return str(metaphlan_output)
        except subprocess.CalledProcessError as e:
            logger.error(f"MetaPhlAn2 failed for {sample_id}: {e.stderr}")
            raise
    
    def extract_marker_genes(self, sample_id: str, metaphlan_output: str) -> str:
        """
        Extract strain-level marker genes from MetaPhlAn2 output.
        
        This extracts markers from the SAM file generated by MetaPhlAn2.
        For production use, consider using StrainPhlAn for more accurate
        strain-level marker extraction.
        
        Args:
            sample_id: Sample identifier
            metaphlan_output: Path to MetaPhlAn2 output
            
        Returns:
            Path to marker gene profile file
        """
        logger.info(f"Extracting marker genes for {sample_id}")
        
        marker_output = self.output_dir / 'markers' / f"{sample_id}_markers.txt"
        
        # Try to extract from SAM file first (more accurate)
        sam_file = self.output_dir / 'metaphlan' / f"{sample_id}.sam.bz2"
        if sam_file.exists():
            return self._extract_markers_from_sam(sample_id, str(sam_file), str(marker_output))
        
        # Fallback: Create markers from MetaPhlAn2 species data
        # This is a simplified approach - for production, use StrainPhlAn
        logger.warning(f"SAM file not found for {sample_id}, using simplified marker extraction")
        
        try:
            metaphlan_df = pd.read_csv(
                metaphlan_output,
                sep='\t',
                comment='#',
                names=['clade_name', 'relative_abundance', 'coverage', 'estimated_number_of_reads']
            )
            
            marker_data = {
                'marker_id': [],
                'presence': []
            }
            
            # Extract markers from species with significant abundance
            # In practice, these would be actual genomic marker coordinates
            significant_species = metaphlan_df[
                (metaphlan_df['relative_abundance'] > 0.01) &
                (metaphlan_df['clade_name'].str.contains('s__', na=False))
            ]
            
            for _, row in significant_species.iterrows():
                species_name = row['clade_name']
                # Generate marker IDs in format similar to MVIB paper
                # Format: gi|ref_id|:coordinate (simplified version)
                species_clean = species_name.replace('|', '_').replace(' ', '_')
                # Create multiple markers per species (simplified)
                for i in range(1, 4):  # 3 markers per species as placeholder
                    marker_id = f"gi|{species_clean}|marker_{i}"
                    marker_data['marker_id'].append(marker_id)
                    marker_data['presence'].append(1)
            
            marker_df = pd.DataFrame(marker_data)
            marker_df.to_csv(marker_output, sep='\t', index=False)
            
            logger.info(f"Marker extraction completed for {sample_id}: {len(marker_df)} markers")
            return str(marker_output)
            
        except Exception as e:
            logger.error(f"Error extracting markers for {sample_id}: {e}")
            # Create empty marker file
            pd.DataFrame(columns=['marker_id', 'presence']).to_csv(marker_output, sep='\t', index=False)
            return str(marker_output)
    
    def _extract_markers_from_sam(self, sample_id: str, sam_file: str, output_file: str) -> str:
        """
        Extract markers from SAM file with genomic coordinates (matches IBD format).
        
        IBD marker format: gi|ref_id|:start-end (e.g., gi|104773257|ref|NC_008054.1|:116729-117526)
        
        Args:
            sample_id: Sample identifier
            sam_file: Path to SAM file
            output_file: Path to output marker file
            
        Returns:
            Path to marker file
        """
        import bz2
        import re
        
        markers = {}
        
        def parse_cigar(cigar):
            """Parse CIGAR string to get alignment length."""
            if cigar == '*':
                return 0
            length = 0
            matches = re.findall(r'(\d+)([MIDNSHPX=])', cigar)
            for num, op in matches:
                if op in 'M=X':  # Match, equal, mismatch
                    length += int(num)
            return length
        
        try:
            opener = bz2.open if sam_file.endswith('.bz2') else open
            mode = 'rt' if sam_file.endswith('.bz2') else 'r'
            
            with opener(sam_file, mode) as f:
                for line in f:
                    if line.startswith('@'):
                        continue  # Skip header
                    
                    fields = line.strip().split('\t')
                    if len(fields) < 10:
                        continue
                    
                    # SAM format fields:
                    # fields[2] = reference name (marker gene reference)
                    # fields[3] = position (1-based alignment start)
                    # fields[5] = CIGAR string (alignment details)
                    
                    ref_name = fields[2]
                    position = fields[3] if len(fields) > 3 else None
                    cigar = fields[5] if len(fields) > 5 else '*'
                    
                    if ref_name != '*' and ('gi|' in ref_name or 'ref|' in ref_name):
                        # Extract genomic coordinates
                        if position and position.isdigit():
                            start_pos = int(position)
                            alignment_length = parse_cigar(cigar)
                            end_pos = start_pos + alignment_length - 1 if alignment_length > 0 else start_pos
                            
                            # Format as IBD: gi|ref_id|:start-end
                            marker_id = f"{ref_name}:{start_pos}-{end_pos}"
                        else:
                            # Fallback: use reference name only if position not available
                            marker_id = ref_name
                        
                        markers[marker_id] = 1
            
            # Convert to DataFrame
            marker_df = pd.DataFrame({
                'marker_id': list(markers.keys()),
                'presence': list(markers.values())
            })
            
            marker_df.to_csv(output_file, sep='\t', index=False)
            logger.info(f"Extracted {len(marker_df)} markers from SAM file for {sample_id}")
            
        except Exception as e:
            logger.warning(f"Error reading SAM file: {e}, using fallback method")
            # Fallback to simplified method
            return self.extract_marker_genes(sample_id, sam_file.replace('.sam.bz2', '_metaphlan.txt'))
        
        return output_file
    
    def format_for_mvib(
        self,
        disease_name: str,
        metadata_columns: List[str] = None
    ) -> Tuple[str, str]:
        """
        Format processed data into MVIB expected format.
        
        This matches the exact format expected by MicrobiomeDataset:
        - Tab-separated files
        - First row: header (will be skipped with skiprows=1)
        - First column: row identifiers (sampleID for metadata, species/marker names for features)
        - Columns: Sample IDs (patients)
        - Rows 0-208: Metadata rows
        - Rows 209+: Feature rows
        
        Args:
            disease_name: Name of the disease/dataset
            metadata_columns: List of metadata columns to include
            
        Returns:
            Tuple of (abundance_file, marker_file) paths
        """
        logger.info(f"Formatting data for MVIB: {disease_name}")
        
        if metadata_columns is None:
            # Match the exact metadata structure from IBD dataset
            metadata_columns = [
                'dataset_name', 'sampleID', 'subjectID', 'bodysite',
                'disease', 'age', 'gender', 'country', 'sequencing_technology',
                'pubmedid', 'camp', 'paired_end_insert_size', 'read_length',
                'total_reads', 'matched_reads', 'uniquely_matching_reads',
                'uniquely_matched_reads', 'gene_number',
                'gene_number_for_11m_uniquely_matched_reads',
                'hitchip_probe_number', 'bmi', 'gene_count_class',
                'hitchip_probe_class', '#SampleID'
            ]
        
        # Collect all MetaPhlAn2 outputs
        metaphlan_files = list((self.output_dir / 'metaphlan').glob('*_metaphlan.txt'))
        marker_files = list((self.output_dir / 'markers').glob('*_markers.txt'))
        
        if not metaphlan_files:
            raise RuntimeError(
                "No MetaPhlAn output files found. Process at least one sample before formatting. "
                "Check that FASTQ files exist and pipeline steps completed."
            )
        
        # Build abundance matrix - use only samples that have output files
        abundance_data = {}
        marker_data = {}
        sample_ids = self.samples_df['sampleID'].tolist()
        processed_ids = [s for s in sample_ids if any(s in str(f) for f in metaphlan_files)]
        if not processed_ids:
            processed_ids = [Path(f).name.replace('_metaphlan.txt', '') for f in metaphlan_files]
        
        for sample_id in processed_ids:
            # Find corresponding files
            metaphlan_file = next(
                (f for f in metaphlan_files if sample_id in str(f)),
                None
            )
            marker_file = next(
                (f for f in marker_files if sample_id in str(f)),
                None
            )
            
            if metaphlan_file:
                # Read MetaPhlAn2 output
                try:
                    df = pd.read_csv(
                        metaphlan_file,
                        sep='\t',
                        comment='#',
                        names=['clade_name', 'relative_abundance', 'coverage', 'estimated_number_of_reads']
                    )
                    
                    # Extract species-level abundances (s__ indicates species)
                    species_df = df[df['clade_name'].str.contains('s__', na=False)]
                    for _, row in species_df.iterrows():
                        species = row['clade_name']
                        abundance = float(row['relative_abundance']) if pd.notna(row['relative_abundance']) else 0.0
                        if species not in abundance_data:
                            abundance_data[species] = {}
                        abundance_data[species][sample_id] = abundance
                except Exception as e:
                    logger.warning(f"Error reading MetaPhlAn2 file for {sample_id}: {e}")
            
            if marker_file:
                # Read marker file
                try:
                    df = pd.read_csv(marker_file, sep='\t')
                    if 'marker_id' in df.columns and 'presence' in df.columns:
                        for _, row in df.iterrows():
                            marker_id = row['marker_id']
                            presence = int(row['presence']) if pd.notna(row['presence']) else 0
                            if marker_id not in marker_data:
                                marker_data[marker_id] = {}
                            marker_data[marker_id][sample_id] = presence
                except Exception as e:
                    logger.warning(f"Error reading marker file for {sample_id}: {e}")
        
        # Create feature DataFrames (features as rows, samples as columns)
        abundance_features = pd.DataFrame(abundance_data).T
        abundance_features.columns = processed_ids
        abundance_features = abundance_features.fillna(0.0)
        
        # Ensure all IBD species are included (443 total)
        # Load reference species list from IBD dataset
        try:
            ibd_ab_ref = pd.read_csv(
                Path(__file__).parent.parent / 'data' / 'default' / 'abundance' / 'abundance_IBD.txt',
                sep='\t', skiprows=1, low_memory=False
            )
            ibd_ab_ref_indexed = ibd_ab_ref.set_index('sampleID')
            ibd_ab_ref_features = ibd_ab_ref_indexed.iloc[209:, :]
            reference_species = [r for r in ibd_ab_ref_features.index if str(r).startswith('k__')]
            
            # Add missing species with zero abundance
            missing_species = set(reference_species) - set(abundance_features.index)
            if missing_species:
                missing_df = pd.DataFrame(0.0, index=list(missing_species), columns=processed_ids)
                abundance_features = pd.concat([abundance_features, missing_df])
                # Reorder to match reference order
                abundance_features = abundance_features.reindex(reference_species)
                logger.info(f"Added {len(missing_species)} missing species with zero abundance")
        except Exception as e:
            logger.warning(f"Could not load IBD reference species list: {e}")
        
        marker_features = pd.DataFrame(marker_data).T
        marker_features.columns = processed_ids
        marker_features = marker_features.fillna(0)
        
        # Ensure all IBD markers are included (91,756 total)
        # Load reference marker list from IBD dataset
        try:
            ibd_mk_ref = pd.read_csv(
                Path(__file__).parent.parent / 'data' / 'default' / 'marker' / 'marker_IBD.txt',
                sep='\t', skiprows=1, low_memory=False
            )
            ibd_mk_ref_indexed = ibd_mk_ref.set_index('sampleID')
            ibd_mk_ref_features = ibd_mk_ref_indexed.iloc[209:, :]
            reference_markers = [r for r in ibd_mk_ref_features.index if 'gi|' in str(r)]
            
            # Add missing markers with zero presence
            missing_markers = set(reference_markers) - set(marker_features.index)
            if missing_markers:
                missing_df = pd.DataFrame(0, index=list(missing_markers), columns=processed_ids)
                marker_features = pd.concat([marker_features, missing_df])
                # Reorder to match reference order
                marker_features = marker_features.reindex(reference_markers)
                logger.info(f"Added {len(missing_markers)} missing markers with zero presence")
        except Exception as e:
            logger.warning(f"Could not load IBD reference marker list: {e}")
        
        # Build metadata DataFrame
        # Structure: metadata rows as index, samples as columns
        metadata_data = {}
        for col in metadata_columns:
            if col == 'sampleID':
                metadata_data[col] = processed_ids
            elif col == 'dataset_name':
                metadata_data[col] = [disease_name] * len(processed_ids)
            elif col == 'sequencing_technology':
                metadata_data[col] = ['Illumina'] * len(processed_ids)
            elif col == 'pubmedid':
                metadata_data[col] = ['nd'] * len(processed_ids)  # Will be filled if available
            elif col == '#SampleID':
                metadata_data[col] = ['Metaphlan2_Analysis'] * len(processed_ids)
            elif col in self.samples_df.columns:
                # Align to processed_ids from sample sheet
                sid_to_val = dict(zip(self.samples_df['sampleID'], self.samples_df[col]))
                metadata_data[col] = [sid_to_val.get(s, 'nd') for s in processed_ids]
            else:
                metadata_data[col] = ['nd'] * len(processed_ids)
        
        # Create metadata DataFrame: rows = metadata fields, columns = samples
        # Transpose so each metadata field becomes a row
        metadata_df = pd.DataFrame(metadata_data, index=processed_ids).T
        metadata_df.index.name = 'sampleID'
        
        # Remove 'dataset_name' from metadata rows (it's only in header, not counted after set_index)
        if 'dataset_name' in metadata_df.index:
            metadata_df = metadata_df.drop('dataset_name')
        
        # Ensure 'sampleID' row is first (it will become column names after skiprows=1)
        if 'sampleID' in metadata_df.index:
            current_order = list(metadata_df.index)
            if current_order.index('sampleID') != 0:
                new_order = ['sampleID'] + [r for r in current_order if r != 'sampleID']
                metadata_df = metadata_df.reindex(new_order)
        
        # MVIB expects exactly 209 metadata rows AFTER set_index('sampleID')
        # After set_index, the 'sampleID' column values become the index
        # The 'sampleID' row provides column names, so it's NOT in the index
        # So we need exactly 209 rows in metadata_df (including sampleID row = 210 total)
        # But wait: after set_index, sampleID row values become column names
        # So if we have 209 rows including sampleID, after set_index we get 208 index rows
        # We need 210 rows total (1 sampleID + 209 metadata) to get 209 index rows
        
        current_metadata_rows = len(metadata_df)
        target_rows = 210  # 1 sampleID row + 209 metadata rows = 209 index rows after set_index
        if current_metadata_rows < target_rows:
            padding_rows = target_rows - current_metadata_rows
            padding_data = {col: ['nd'] * padding_rows for col in metadata_df.columns}
            padding_df = pd.DataFrame(padding_data, 
                                     index=[f'padding_{i}' for i in range(padding_rows)])
            metadata_df = pd.concat([metadata_df, padding_df])
            logger.info(f"Padded metadata rows from {current_metadata_rows} to {target_rows} (209 after set_index)")
        elif current_metadata_rows > target_rows:
            metadata_df = metadata_df.iloc[:target_rows]
            logger.info(f"Trimmed metadata rows from {current_metadata_rows} to {target_rows}")
        
        # This matches MVIB format: rows 0-208 are metadata, rows 209+ are features
        # Both should have same column structure (samples)
        abundance_formatted = pd.concat([metadata_df, abundance_features])
        marker_formatted = pd.concat([metadata_df, marker_features])
        
        # Reset index to make it a column (row identifiers)
        abundance_formatted = abundance_formatted.reset_index()
        abundance_formatted = abundance_formatted.rename(columns={'index': 'row_id'})
        
        marker_formatted = marker_formatted.reset_index()
        marker_formatted = marker_formatted.rename(columns={'index': 'row_id'})
        
        # MVIB expects: 
        # - Header row (line 0): first column is metadata name (e.g., 'dataset_name'), rest are sample IDs
        # - Data rows: first column is row identifier (metadata name or species name), rest are values
        # - After skiprows=1: pandas reads columns=['sampleID', 'SAMPLE1', 'SAMPLE2', ...]
        #   where 'sampleID' comes from the first data row's first column value
        
        # Reorder columns: row_id first, then sample IDs
        col_order = ['row_id'] + processed_ids
        abundance_formatted = abundance_formatted[col_order]
        marker_formatted = marker_formatted[col_order]
        
        # Create header row: first column should be a metadata name (like 'dataset_name'), 
        # rest should be sample IDs repeated (or use first metadata value)
        # Actually, looking at original: header is 'dataset_name' + sample IDs repeated
        # But after skiprows=1, the first data row becomes column names
        # So we need: first data row's first column = 'sampleID', rest = actual sample IDs
        
        # The header row should have: first column = first metadata name, rest = sample IDs
        # But MVIB skips it with skiprows=1, so it doesn't matter much
        # What matters: first data row must have 'sampleID' as first value
        
        # Ensure 'sampleID' row exists and is first data row
        # Check if 'sampleID' is already the first row
        if abundance_formatted.iloc[0]['row_id'] != 'sampleID':
            # Find sampleID row and move it to first position
            sampleid_idx = abundance_formatted[abundance_formatted['row_id'] == 'sampleID'].index
            if len(sampleid_idx) > 0:
                # Remove it from current position and insert at beginning
                sampleid_row = abundance_formatted.loc[sampleid_idx[0]]
                abundance_formatted = abundance_formatted.drop(sampleid_idx[0])
                abundance_formatted = pd.concat([pd.DataFrame([sampleid_row]), abundance_formatted]).reset_index(drop=True)
                # Do same for marker
                sampleid_idx_mk = marker_formatted[marker_formatted['row_id'] == 'sampleID'].index
                if len(sampleid_idx_mk) > 0:
                    sampleid_row_mk = marker_formatted.loc[sampleid_idx_mk[0]]
                    marker_formatted = marker_formatted.drop(sampleid_idx_mk[0])
                    marker_formatted = pd.concat([pd.DataFrame([sampleid_row_mk]), marker_formatted]).reset_index(drop=True)
        
        # Rename 'row_id' column back to match first row value for header
        # Actually, we want the first column to be unnamed in the header, but named in data
        # Let's set the column name to match what MVIB expects after skiprows=1
        # After skiprows=1, first data row becomes column names, so first column name should be 'sampleID'
        abundance_formatted.columns.values[0] = 'sampleID'
        marker_formatted.columns.values[0] = 'sampleID'
        
        # Save formatted files
        output_dir = self.output_dir / 'formatted'
        abundance_file = output_dir / f'abundance_{disease_name}.txt'
        marker_file = output_dir / f'marker_{disease_name}.txt'
        
        # Save with proper format: 
        # - Header row: first column = first metadata name (e.g., 'dataset_name'), rest = sample IDs
        # - Data rows: first column = row identifier, rest = values
        # MVIB uses skiprows=1, so header is skipped and first data row becomes column names
        
        # Create header row: use first metadata name + sample IDs
        header_row_ab = ['dataset_name'] + processed_ids
        header_row_mk = ['dataset_name'] + processed_ids
        
        # Write header + data
        with open(abundance_file, 'w') as f:
            f.write('\t'.join(header_row_ab) + '\n')
            abundance_formatted.to_csv(f, sep='\t', index=False, header=False)
        
        with open(marker_file, 'w') as f:
            f.write('\t'.join(header_row_mk) + '\n')
            marker_formatted.to_csv(f, sep='\t', index=False, header=False)
        
        logger.info(f"Formatted files saved:")
        logger.info(f"  Abundance: {abundance_file}")
        logger.info(f"    Shape: {abundance_formatted.shape}")
        logger.info(f"    Metadata rows: {len(metadata_columns)}")
        logger.info(f"    Feature rows: {len(abundance_features)}")
        logger.info(f"  Marker: {marker_file}")
        logger.info(f"    Shape: {marker_formatted.shape}")
        logger.info(f"    Metadata rows: {len(metadata_columns)}")
        logger.info(f"    Feature rows: {len(marker_features)}")
        
        return str(abundance_file), str(marker_file)
    
    def _process_single_sample(
        self, sample_id: str, skip_qc: bool
    ) -> Tuple[str, bool]:
        """
        Process one sample: QC -> trim -> MetaPhlAn -> markers.
        Returns (sample_id, success).
        """
        forward_pattern = f"**/*{sample_id}*R1*.fastq*"
        reverse_pattern = f"**/*{sample_id}*R2*.fastq*"
        forward_files = sorted(self.input_dir.glob(forward_pattern))
        reverse_files = sorted(self.input_dir.glob(reverse_pattern))
        
        if not forward_files:
            logger.warning(f"No FASTQ files found for {sample_id}, skipping")
            return (sample_id, False)
        
        forward_reads = str(forward_files[0])
        reverse_reads = str(reverse_files[0]) if reverse_files else None
        
        try:
            if not skip_qc:
                self.run_quality_control(
                    sample_id, [forward_reads] + ([reverse_reads] if reverse_reads else [])
                )
            trimmed_forward, trimmed_reverse = self.trim_reads(
                sample_id, forward_reads, reverse_reads
            )
            metaphlan_output = self.run_metaphlan2(
                sample_id, trimmed_forward, trimmed_reverse
            )
            self.extract_marker_genes(sample_id, metaphlan_output)
            return (sample_id, True)
        except Exception as e:
            logger.error(f"Failed to process {sample_id}: {e}")
            return (sample_id, False)
    
    def run_pipeline(
        self,
        disease_name: str,
        skip_qc: bool = False,
        jobs: int = 1,
        max_samples: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            disease_name: Name of the disease/dataset
            skip_qc: Skip quality control step
            jobs: Number of samples to process in parallel (default: 1)
            max_samples: Process only first N samples (None = all)
            
        Returns:
            Dictionary with paths to final output files
        """
        logger.info("="*80)
        logger.info("Starting Shotgun Sequencing Preprocessing Pipeline")
        logger.info("="*80)
        if jobs > 1:
            logger.info(f"Parallel mode: {jobs} samples at a time")
        
        # Check dependencies
        if not self.check_dependencies():
            raise RuntimeError("Missing required dependencies")
        
        sample_ids = self.samples_df['sampleID'].tolist()
        if max_samples is not None:
            sample_ids = sample_ids[:max_samples]
            logger.info(f"Limiting to first {max_samples} sample(s): {sample_ids}")
        
        if jobs <= 1:
            # Sequential
            for sample_id in sample_ids:
                logger.info(f"\nProcessing sample: {sample_id}")
                self._process_single_sample(sample_id, skip_qc)
        else:
            # Parallel: ThreadPoolExecutor (subprocess calls release GIL)
            completed = 0
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(self._process_single_sample, sid, skip_qc): sid
                    for sid in sample_ids
                }
                for future in as_completed(futures):
                    sample_id, success = future.result()
                    completed += 1
                    status = "OK" if success else "FAILED"
                    logger.info(f"[{completed}/{len(sample_ids)}] {sample_id}: {status}")
        
        # Step 5: Format for MVIB (runs after all samples)
        abundance_file, marker_file = self.format_for_mvib(disease_name)
        
        logger.info("="*80)
        logger.info("Preprocessing Pipeline Completed Successfully")
        logger.info("="*80)
        
        return {
            'abundance_file': abundance_file,
            'marker_file': marker_file,
            'output_dir': str(self.output_dir)
        }


def create_sample_sheet_template(output_path: str):
    """Create a template sample sheet CSV file."""
    template = pd.DataFrame({
        'sampleID': ['SAMPLE_001', 'SAMPLE_002'],
        'subjectID': ['SUBJECT_001', 'SUBJECT_002'],
        'bodysite': ['stool', 'stool'],
        'disease': ['n', 'disease_name'],
        'age': ['30', '45'],
        'gender': ['M', 'F'],
        'country': ['USA', 'USA'],
        'dataset_name': ['MyDataset', 'MyDataset']
    })
    template.to_csv(output_path, index=False)
    logger.info(f"Sample sheet template created: {output_path}")


def create_sample_sheet_from_fastq(fastq_dir: str, output_path: str) -> int:
    """
    Scan fastq_downloads and create a sample sheet from discovered samples.
    Handles both: subdirs (MH0011/MH0011_R1.fastq.gz) and root files (ERR209969.fastq.gz).
    
    Returns:
        Number of samples found
    """
    fastq_path = Path(fastq_dir)
    if not fastq_path.exists():
        logger.error(f"Directory not found: {fastq_dir}")
        return 0
    
    samples = {}
    for f in fastq_path.rglob('*.fastq*'):
        if f.suffix in {'.gz', '.fastq', '.fq'}:
            name = f.stem.replace('.fastq', '').replace('.fq', '')
            # Extract sample ID: MH0011_R1 -> MH0011, ERR209969 -> ERR209969
            if '_R1' in name or '_R2' in name:
                sample_id = name.rsplit('_R', 1)[0]
            else:
                sample_id = name
            samples[sample_id] = samples.get(sample_id, {})
    
    if not samples:
        logger.warning(f"No FASTQ files found in {fastq_dir}")
        return 0
    
    df = pd.DataFrame({
        'sampleID': sorted(samples.keys()),
        'subjectID': sorted(samples.keys()),
        'bodysite': ['stool'] * len(samples),
        'disease': ['nd'] * len(samples),
        'age': ['nd'] * len(samples),
        'gender': ['nd'] * len(samples),
        'country': ['nd'] * len(samples),
        'dataset_name': ['ShotgunData'] * len(samples)
    })
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample sheet with {len(samples)} samples: {output_path}")
    return len(samples)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess shotgun metagenomic sequencing data for MVIB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses defaults: fastq_downloads, processed_data)
  python preprocess_shotgun_data.py --disease IBD

  # Parallel: 8 samples at once, 32 threads each (~4-5h on 256-core HPC)
  python preprocess_shotgun_data.py --disease IBD --jobs 8 --threads 32

  # Create sample sheet from discovered FASTQ files first
  python preprocess_shotgun_data.py --create-sample-sheet fastq_downloads/sample_sheet.csv
  python preprocess_shotgun_data.py --sample-sheet fastq_downloads/sample_sheet.csv --disease IBD

  # Custom paths
  python preprocess_shotgun_data.py \\
      --input-dir /home/80029644/khai_project/MVIB/fastq_downloads \\
      --output-dir /home/80029644/khai_project/MVIB/processed_data \\
      --sample-sheet samples.csv \\
      --disease MyDisease

  # Create sample sheet template
  python preprocess_shotgun_data.py --create-template samples_template.csv
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(DEFAULT_FASTQ_DIR),
        help=f'Directory containing raw FASTQ files (default: {DEFAULT_FASTQ_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f'Output directory for processed files (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--sample-sheet',
        type=str,
        default=str(DEFAULT_SAMPLE_SHEET),
        help=f'CSV file with sample metadata (default: {DEFAULT_SAMPLE_SHEET})'
    )
    parser.add_argument(
        '--disease',
        type=str,
        help='Name of the disease/dataset'
    )
    parser.add_argument(
        '--metaphlan-db',
        type=str,
        default=None,
        help='Path to MetaPhlAn2 database (optional)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=8,
        help='Number of threads for parallel processing (default: 8)'
    )
    parser.add_argument(
        '--min-read-length',
        type=int,
        default=50,
        help='Minimum read length after trimming (default: 50)'
    )
    parser.add_argument(
        '--quality-threshold',
        type=int,
        default=20,
        help='Quality score threshold for trimming (default: 20)'
    )
    parser.add_argument(
        '--skip-qc',
        action='store_true',
        help='Skip quality control step'
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=1,
        metavar='N',
        help='Number of samples to process in parallel (default: 1). Use 4-8 on HPC for ~4-6h total.'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        metavar='N',
        help='Process only first N samples (for testing). Default: all samples.'
    )
    parser.add_argument(
        '--create-template',
        type=str,
        metavar='PATH',
        help='Create a sample sheet template and exit'
    )
    parser.add_argument(
        '--create-sample-sheet',
        type=str,
        metavar='PATH',
        help='Scan fastq_downloads and create sample sheet from discovered FASTQ files'
    )
    
    args = parser.parse_args()
    
    if args.create_template:
        create_sample_sheet_template(args.create_template)
        return
    
    if args.create_sample_sheet:
        fastq_dir = args.input_dir if args.input_dir else str(DEFAULT_FASTQ_DIR)
        n = create_sample_sheet_from_fastq(fastq_dir, args.create_sample_sheet)
        if n > 0:
            logger.info(f"Run pipeline with: --sample-sheet {args.create_sample_sheet} --disease IBD")
        return
    
    if not args.disease:
        parser.error("--disease is required")
    
    # Validate paths exist
    if not Path(args.input_dir).exists():
        logger.warning(f"Input directory does not exist: {args.input_dir}")
    if not Path(args.sample_sheet).exists():
        logger.warning(f"Sample sheet does not exist: {args.sample_sheet}")
        logger.info("Create one with: python preprocess_shotgun_data.py --create-template samples.csv")
    
    # Initialize preprocessor
    preprocessor = ShotgunPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_sheet=args.sample_sheet,
        metaphlan_db=args.metaphlan_db,
        threads=args.threads,
        min_read_length=args.min_read_length,
        quality_threshold=args.quality_threshold
    )
    
    # Run pipeline
    try:
        results = preprocessor.run_pipeline(
            args.disease,
            skip_qc=args.skip_qc,
            jobs=args.jobs,
            max_samples=args.max_samples,
        )
        
        # Save summary
        summary = {
            'disease': args.disease,
            'output_files': results,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'threads': args.threads,
                'min_read_length': args.min_read_length,
                'quality_threshold': args.quality_threshold
            }
        }
        
        summary_file = Path(args.output_dir) / 'preprocessing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nPreprocessing summary saved to: {summary_file}")
        logger.info("\nNext steps:")
        logger.info(f"1. Review formatted files in: {results['output_dir']}/formatted/")
        logger.info(f"2. Copy files to MVIB data directory:")
        logger.info(f"   cp {results['abundance_file']} /path/to/MVIB/data/default/abundance/")
        logger.info(f"   cp {results['marker_file']} /path/to/MVIB/data/default/marker/")
        logger.info("3. Use with MVIB dataset class")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
