"""
PlantMedic AI - Main Entry Point
This file serves as the main entry point for Streamlit Cloud deployment
"""
import streamlit as st
import subprocess
import sys
import os

# Set the main app file
APP_FILE = "app.py"

if __name__ == "__main__":
    # Ensure we're in the right directory
    if os.path.exists(APP_FILE):
        # Import and run the main app
        import app
    else:
        st.error(f"❌ Main application file '{APP_FILE}' not found!")
        st.info("Please ensure app.py is in the root directory.")
