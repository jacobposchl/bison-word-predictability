"""
EAF file processing utilities.

This module handles loading and processing ELAN Annotation Format (EAF) files
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pympi
from ..core.text_cleaning import has_content


def parse_participant_info(tier_name: str) -> Dict[str, str]:
    """
    Extract participant ID and group from tier name.
    
    Tier naming conventions:
    - ACHE*: Heritage speakers
    - ACI*: Immersed speakers
    - ACH*: Homeland speakers
    
    Args:
        tier_name: Name of the tier (e.g., 'ACH2004', 'ACHE2001', 'ACI2003')
        
    Returns:
        Dictionary with 'participant_id', 'group', and 'group_code' keys
    """
    if tier_name.startswith('ACHE'):
        return {
            'participant_id': tier_name,
            'group': 'Heritage',
            'group_code': 'HE'
        }
    elif tier_name.startswith('ACI'):
        return {
            'participant_id': tier_name,
            'group': 'Immersed',
            'group_code': 'I'
        }
    elif tier_name.startswith('ACH'):
        return {
            'participant_id': tier_name,
            'group': 'Homeland',
            'group_code': 'H'
        }
    else:
        return {
            'participant_id': tier_name,
            'group': 'Unknown',
            'group_code': 'U'
        }


def load_eaf_file(file_path: str) -> pympi.Eaf:
    """
    Load and validate an EAF file.
    
    Args:
        file_path: Path to the EAF file
        
    Returns:
        EAF object from pympi
        
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EAF file not found: {file_path}")
    
    try:
        eaf = pympi.Eaf(file_path)
        return eaf
    except Exception as e:
        raise ValueError(f"Error loading EAF file {file_path}: {e}")


def get_main_tier(eaf: pympi.Eaf) -> Optional[str]:
    """
    Extract the main participant tier name from an EAF file.
    
    Looks for tiers that start with ACH, ACHE, or ACI and don't have hyphens.
    
    Args:
        eaf: EAF object
        
    Returns:
        Main tier name, or None if not found
    """

    tier_names = eaf.get_tier_names()
    participant_tiers = [
        t for t in tier_names
        if (t.startswith('ACH') or t.startswith('ACI'))
        and '-' not in t
    ]
    
    if not participant_tiers:
        return None
    
    return participant_tiers[0]


def get_all_eaf_files(data_path: str) -> List[str]:
    """
    Scan data directory for all EAF files.
    
    Args:
        data_path: Path to directory containing EAF files
        
    Returns:
        Sorted list of EAF file names
        
    Raises:
        FileNotFoundError: If the data path doesn't exist
    """
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path is not a directory: {data_path}")
    
    eaf_files = sorted([
        f for f in os.listdir(data_path)
        if f.endswith('.eaf')
    ])
    
    return eaf_files

