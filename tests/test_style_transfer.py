import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from style_transfer import TextStyleTransfer
from database import StyleTransferDatabase

class TestStyleTransfer:
    def test_initialization(self):
        """Test that TextStyleTransfer initializes correctly."""
        # This is a basic test - in practice, you'd mock the model loading
        pass
    
    def test_database_initialization(self):
        """Test database initialization."""
        db = StyleTransferDatabase(":memory:")  # Use in-memory database for testing
        assert db is not None
    
    def test_config_loading(self):
        """Test configuration loading."""
        # Test with default config
        transfer = TextStyleTransfer()
        assert transfer.config is not None

if __name__ == "__main__":
    pytest.main([__file__])
