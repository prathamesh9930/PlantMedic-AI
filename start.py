"""
PlantMedic AI - Startup Script
Run this script to start the application with proper error handling
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all required packages are installed"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found!")
        return False
    
    try:
        # Try importing key packages
        import streamlit
        import tensorflow
        import plotly
        import pandas
        logger.info("All required packages are available")
        return True
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.info("Installing requirements...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            logger.info("Requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to install requirements")
            return False

def check_model_files():
    """Check if model files exist"""
    model_file = Path("plant_disease_model.h5")
    class_names_file = Path("class_names.pkl")
    
    if not model_file.exists():
        logger.error("Model file 'plant_disease_model.h5' not found!")
        logger.info("Please ensure the model file is in the project directory")
        return False
    
    if not class_names_file.exists():
        logger.error("Class names file 'class_names.pkl' not found!")
        logger.info("Please ensure the class names file is in the project directory")
        return False
    
    logger.info("Model files found")
    return True

def start_application():
    """Start the Streamlit application"""
    try:
        logger.info("Starting PlantMedic AI...")
        logger.info("The application will open in your default web browser")
        logger.info("Press Ctrl+C to stop the application")
        
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.serverAddress", "localhost"
        ])
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Error starting application: {e}")

def main():
    """Main startup function"""
    logger.info("� PlantMedic AI - Startup Script")
    logger.info("=" * 50)
    
    # Check requirements
    if not check_requirements():
        logger.error("Requirements check failed. Please install the required packages manually.")
        return
    
    # Check model files
    if not check_model_files():
        logger.error("Model files check failed. Please ensure model files are available.")
        return
    
    # Start application
    start_application()

if __name__ == "__main__":
    main()
