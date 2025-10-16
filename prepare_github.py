#!/usr/bin/env python3
"""
GitHub preparation script for Text Style Transfer project.
Creates necessary files and structure for GitHub repository.
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime

def create_license():
    """Create MIT License file."""
    license_content = """MIT License

Copyright (c) 2024 Text Style Transfer Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    with open("LICENSE", "w") as f:
        f.write(license_content)
    print("✅ Created LICENSE file")

def create_contributing():
    """Create CONTRIBUTING.md file."""
    contributing_content = """# Contributing to Text Style Transfer

Thank you for your interest in contributing to the Text Style Transfer project!

## How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs or request features
- Provide detailed information about the issue
- Include system information and error messages when applicable

### Submitting Pull Requests
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Submit a pull request with a clear description

### Development Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m pytest`
4. Run the demo: `python 0200.py`

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular

### Areas for Contribution
- New style transfer models
- Additional evaluation metrics
- UI/UX improvements
- Documentation improvements
- Performance optimizations
- Test coverage

## Questions?
Feel free to open an issue for any questions or discussions.
"""
    
    with open("CONTRIBUTING.md", "w") as f:
        f.write(contributing_content)
    print("✅ Created CONTRIBUTING.md file")

def create_changelog():
    """Create CHANGELOG.md file."""
    changelog_content = f"""# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - {datetime.now().strftime('%Y-%m-%d')}

### Added
- Initial release of Text Style Transfer system
- Support for multiple AI models (T5, GPT-2, BART)
- Comprehensive style categories (formal/informal, positive/negative, modern/Shakespearean)
- Real-time evaluation metrics (BLEU, ROUGE, BERT Score, Semantic Similarity)
- Modern Streamlit web interface
- SQLite database for sample texts and transfer history
- Batch processing capabilities
- YAML-based configuration system
- Comprehensive documentation and examples

### Features
- Interactive web UI with visualizations
- Model selection and parameter tuning
- Transfer history and statistics
- Performance monitoring and logging
- Easy setup and installation scripts

### Technical Details
- Built with modern Python libraries (Transformers, Streamlit, Plotly)
- GPU acceleration support
- Comprehensive error handling
- Modular and extensible architecture
"""
    
    with open("CHANGELOG.md", "w") as f:
        f.write(changelog_content)
    print("✅ Created CHANGELOG.md file")

def create_dockerfile():
    """Create Dockerfile for containerization."""
    dockerfile_content = """# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs models

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    print("✅ Created Dockerfile")

def create_docker_compose():
    """Create docker-compose.yml file."""
    compose_content = """version: '3.8'

services:
  text-style-transfer:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(compose_content)
    print("✅ Created docker-compose.yml file")

def create_github_workflows():
    """Create GitHub Actions workflows."""
    workflows_dir = Path(".github/workflows")
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # CI/CD workflow
    ci_workflow = """name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Test with pytest
      run: |
        pip install pytest
        python -m pytest tests/ -v
    
    - name: Test basic functionality
      run: |
        python -c "from style_transfer import TextStyleTransfer; print('Import test passed')"
"""
    
    with open(".github/workflows/ci.yml", "w") as f:
        f.write(ci_workflow)
    print("✅ Created GitHub Actions CI workflow")

def create_tests():
    """Create basic test files."""
    tests_dir = Path("tests")
    tests_dir.mkdir(exist_ok=True)
    
    # Test file
    test_content = """import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from style_transfer import TextStyleTransfer
from database import StyleTransferDatabase

class TestStyleTransfer:
    def test_initialization(self):
        \"\"\"Test that TextStyleTransfer initializes correctly.\"\"\"
        # This is a basic test - in practice, you'd mock the model loading
        pass
    
    def test_database_initialization(self):
        \"\"\"Test database initialization.\"\"\"
        db = StyleTransferDatabase(":memory:")  # Use in-memory database for testing
        assert db is not None
    
    def test_config_loading(self):
        \"\"\"Test configuration loading.\"\"\"
        # Test with default config
        transfer = TextStyleTransfer()
        assert transfer.config is not None

if __name__ == "__main__":
    pytest.main([__file__])
"""
    
    with open("tests/test_style_transfer.py", "w") as f:
        f.write(test_content)
    print("✅ Created test files")

def create_project_structure():
    """Create final project structure."""
    print("📁 Final project structure:")
    
    files = [
        "0200.py",
        "style_transfer.py", 
        "database.py",
        "app.py",
        "config.yaml",
        "requirements.txt",
        "setup.py",
        "README.md",
        ".gitignore",
        "LICENSE",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "Dockerfile",
        "docker-compose.yml",
        ".github/workflows/ci.yml",
        "tests/test_style_transfer.py"
    ]
    
    for file in files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (missing)")

def main():
    """Main GitHub preparation function."""
    print("🚀 GitHub Preparation for Text Style Transfer")
    print("=" * 50)
    
    # Create all necessary files
    create_license()
    create_contributing()
    create_changelog()
    create_dockerfile()
    create_docker_compose()
    create_github_workflows()
    create_tests()
    
    # Show final structure
    create_project_structure()
    
    print("\n🎉 GitHub preparation completed!")
    print("\n📋 Next steps for GitHub:")
    print("  1. Initialize git repository: git init")
    print("  2. Add all files: git add .")
    print("  3. Initial commit: git commit -m 'Initial commit'")
    print("  4. Create GitHub repository")
    print("  5. Push to GitHub: git push -u origin main")
    print("\n🐳 Docker deployment:")
    print("  - Build: docker build -t text-style-transfer .")
    print("  - Run: docker-compose up")
    print("\n📚 Documentation:")
    print("  - README.md: Complete project documentation")
    print("  - CONTRIBUTING.md: Contribution guidelines")
    print("  - CHANGELOG.md: Version history")

if __name__ == "__main__":
    main()
