#!/usr/bin/env python3
"""BERTopic and enhanced NLP dependencies installation script"""

import subprocess
import sys
from pathlib import Path

# Add src to path for logging imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.logging_config import get_logger, setup_cli_logging

# Set up logging
setup_cli_logging(verbose=True)
logger = get_logger(__name__)

def install_bertopic_dependencies():
    """Install BERTopic and related dependencies for enhanced topic modeling"""
    try:
        logger.info("Installing enhanced topic modeling dependencies...")
        
        # Core BERTopic packages
        packages = [
            "bertopic==0.16.0",
            "umap-learn==0.5.3", 
            "hdbscan==0.8.33",
            "transformers==4.36.0",
            "torch==2.1.0",
            # Additional useful packages
            "datasets",  # For accessing pre-trained datasets
            "scikit-learn>=1.3.0"  # Ensure latest sklearn
        ]
        
        for package in packages:
            logger.info(f"Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully installed {package}")
            else:
                logger.error(f"❌ Error installing {package}: {result.stderr}")
                return False
        
        # Verify installation
        logger.info("Verifying BERTopic installation...")
        
        try:
            import bertopic
            import umap
            import hdbscan
            import transformers
            
            logger.info("✅ All BERTopic dependencies installed successfully!")
            logger.info(f"BERTopic version: {bertopic.__version__}")
            logger.info(f"UMAP version: {umap.__version__}")
            logger.info(f"HDBSCAN version: {hdbscan.__version__}")
            logger.info(f"Transformers version: {transformers.__version__}")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ Import verification failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during installation: {e}")
        return False

def download_sentence_transformer_model():
    """Download the default sentence transformer model for faster startup"""
    try:
        logger.info("Pre-downloading sentence transformer model...")
        
        from sentence_transformers import SentenceTransformer
        
        # Download the model we use in TopicAnalyzer
        model = SentenceTransformer('all-mpnet-base-v2')
        
        logger.info("✅ Sentence transformer model downloaded successfully!")
        logger.info(f"Model max sequence length: {model.max_seq_length}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = install_bertopic_dependencies()
    
    if success:
        logger.info("Attempting to download sentence transformer model...")
        download_sentence_transformer_model()
        
        logger.info("🎉 Enhanced topic modeling setup complete!")
        logger.info("You can now use:")
        logger.info("  poetry run pepino analyze topics -d 365  # for much better AI/GenAI topic extraction")
        
    else:
        logger.error("❌ Installation failed. Please check the errors above.")
        sys.exit(1) 