#!/usr/bin/env python3
"""spaCy model installation script"""

import subprocess
import sys
from pathlib import Path

# Add src to path for logging imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.logging_config import get_logger, setup_cli_logging

# Set up logging
setup_cli_logging(verbose=True)
logger = get_logger(__name__)

def install_spacy_model():
    """Install the required spaCy English model"""
    try:
        logger.info("Installing spaCy English model...")
        result = subprocess.run([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ spaCy English model installed successfully!")
        else:
            logger.error(f"❌ Error installing spaCy model: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    install_spacy_model() 