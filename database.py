import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any
import os

class StyleTransferDatabase:
    """Mock database for storing text style transfer samples and results."""
    
    def __init__(self, db_path: str = "data/style_transfer_samples.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create sample texts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sample_texts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    category TEXT,
                    style_type TEXT,
                    sentiment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create style categories table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS style_categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    prompt_template TEXT,
                    examples TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create transfer results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transfer_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_text TEXT NOT NULL,
                    transferred_text TEXT NOT NULL,
                    style_category TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    confidence_score REAL,
                    evaluation_metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            self.populate_sample_data()
    
    def populate_sample_data(self):
        """Populate the database with sample data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Sample texts for different categories
            sample_texts = [
                # Formal texts
                ("The presentation was highly professional and informative.", "business", "formal", "positive"),
                ("I am extremely disappointed with the product I received.", "business", "formal", "negative"),
                ("We would like to express our gratitude for your assistance.", "business", "formal", "positive"),
                ("The implementation of this solution requires careful consideration.", "technical", "formal", "neutral"),
                
                # Informal texts
                ("This is awesome! I totally love it!", "casual", "informal", "positive"),
                ("I'm totally not happy with this.", "casual", "informal", "negative"),
                ("Hey, thanks a lot for helping me out!", "casual", "informal", "positive"),
                ("This is pretty cool, I guess.", "casual", "informal", "neutral"),
                
                # Shakespearean texts
                ("Thou art most fair, my dearest love.", "literary", "shakespearean", "positive"),
                ("Hark! What light through yonder window breaks?", "literary", "shakespearean", "neutral"),
                ("Alas, poor Yorick! I knew him well.", "literary", "shakespearean", "negative"),
                ("To be or not to be, that is the question.", "literary", "shakespearean", "neutral"),
                
                # Modern texts
                ("I love you very much.", "personal", "modern", "positive"),
                ("This is a beautiful day.", "personal", "modern", "positive"),
                ("I don't understand what you mean.", "personal", "modern", "negative"),
                ("The weather is nice today.", "personal", "modern", "neutral"),
            ]
            
            cursor.executemany('''
                INSERT OR IGNORE INTO sample_texts (text, category, style_type, sentiment)
                VALUES (?, ?, ?, ?)
            ''', sample_texts)
            
            # Style categories
            style_categories = [
                ("formal_to_informal", "Convert formal language to casual/informal", 
                 "Rewrite this formal text in a casual, informal style: {text}",
                 json.dumps(["The presentation was highly professional and informative.", "I am extremely disappointed with the product I received."])),
                
                ("informal_to_formal", "Convert casual language to formal/professional",
                 "Rewrite this casual text in a formal, professional style: {text}",
                 json.dumps(["This is awesome!", "I'm totally not happy with this."])),
                
                ("positive_to_negative", "Change positive sentiment to negative",
                 "Rewrite this positive text with negative sentiment: {text}",
                 json.dumps(["I love this product!", "This service is excellent."])),
                
                ("negative_to_positive", "Change negative sentiment to positive",
                 "Rewrite this negative text with positive sentiment: {text}",
                 json.dumps(["This product is terrible.", "I hate this service."])),
                
                ("modern_to_shakespearean", "Convert modern English to Shakespearean style",
                 "Rewrite this modern text in Shakespearean style: {text}",
                 json.dumps(["I love you very much.", "This is a beautiful day."])),
                
                ("shakespearean_to_modern", "Convert Shakespearean style to modern English",
                 "Rewrite this Shakespearean text in modern English: {text}",
                 json.dumps(["Thou art most fair.", "Hark! What light through yonder window breaks?"])),
            ]
            
            cursor.executemany('''
                INSERT OR IGNORE INTO style_categories (name, description, prompt_template, examples)
                VALUES (?, ?, ?, ?)
            ''', style_categories)
            
            conn.commit()
    
    def get_sample_texts(self, style_type: str = None, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve sample texts from the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM sample_texts WHERE 1=1"
            params = []
            
            if style_type:
                query += " AND style_type = ?"
                params.append(style_type)
            
            if category:
                query += " AND category = ?"
                params.append(category)
            
            query += " ORDER BY RANDOM() LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_style_categories(self) -> List[Dict[str, Any]]:
        """Retrieve all style categories."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM style_categories")
            return [dict(row) for row in cursor.fetchall()]
    
    def save_transfer_result(self, original_text: str, transferred_text: str, 
                           style_category: str, model_name: str, 
                           confidence_score: float = None, 
                           evaluation_metrics: Dict[str, Any] = None):
        """Save a style transfer result to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO transfer_results 
                (original_text, transferred_text, style_category, model_name, confidence_score, evaluation_metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (original_text, transferred_text, style_category, model_name, 
                  confidence_score, json.dumps(evaluation_metrics) if evaluation_metrics else None))
            conn.commit()
    
    def get_transfer_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent transfer results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM transfer_results 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

# Example usage
if __name__ == "__main__":
    db = StyleTransferDatabase()
    
    # Get some sample texts
    formal_texts = db.get_sample_texts(style_type="formal", limit=5)
    print("Formal texts:")
    for text in formal_texts:
        print(f"- {text['text']}")
    
    # Get style categories
    categories = db.get_style_categories()
    print("\nStyle categories:")
    for cat in categories:
        print(f"- {cat['name']}: {cat['description']}")
