import streamlit as st
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.streamlit_app import PortfolioUI

def main():
    """Main application entry point"""
    portfolio_ui = PortfolioUI()
    portfolio_ui.run()

if __name__ == "__main__":
    main()