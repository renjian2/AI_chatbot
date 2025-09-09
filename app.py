# app.py

import streamlit as st
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("file_upload_debug.log"),
        logging.StreamHandler()
    ]
)

from ui.main_fixed import run_ui

def main():
    """Main function to run the Streamlit application."""
    run_ui()

if __name__ == "__main__":
    main()