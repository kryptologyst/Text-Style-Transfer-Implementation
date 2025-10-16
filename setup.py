#!/usr/bin/env python3
"""
Setup script for Text Style Transfer project.
Handles installation, configuration, and initial setup.
"""

import os
import sys
import subprocess
import sqlite3
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ pip is not available. Please install pip first.")
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "logs",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def initialize_database():
    """Initialize the database."""
    print("🗄️ Initializing database...")
    
    try:
        from database import StyleTransferDatabase
        db = StyleTransferDatabase()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def test_installation():
    """Test the installation."""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import torch
        import transformers
        import streamlit
        import pandas
        import plotly
        
        print("✅ All required packages imported successfully")
        
        # Test basic functionality
        from style_transfer import TextStyleTransfer
        print("✅ Style transfer module imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def create_sample_config():
    """Create sample configuration if it doesn't exist."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("📝 Creating sample configuration...")
        # The config.yaml should already exist from our earlier creation
        if config_path.exists():
            print("✅ Configuration file already exists")
        else:
            print("⚠️ Configuration file not found, please check config.yaml")
    else:
        print("✅ Configuration file exists")

def main():
    """Main setup function."""
    print("🎭 Text Style Transfer Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        print("Please install manually: pip install -r requirements.txt")
        sys.exit(1)
    
    # Initialize database
    if not initialize_database():
        print("❌ Failed to initialize database")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        sys.exit(1)
    
    # Create sample config
    create_sample_config()
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 Next steps:")
    print("  1. Run the demo: python 0200.py")
    print("  2. Launch web UI: streamlit run app.py")
    print("  3. Check README.md for detailed usage instructions")
    print("\n📚 Available commands:")
    print("  - python 0200.py          # Run basic demo")
    print("  - python style_transfer.py # Run full system")
    print("  - streamlit run app.py    # Launch web interface")

if __name__ == "__main__":
    main()
