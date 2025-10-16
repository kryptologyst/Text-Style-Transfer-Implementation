"""
Project 200: Modern Text Style Transfer Implementation
=====================================================

Description:
Text Style Transfer rewrites a sentence to preserve its meaning while changing its style — 
for example, converting formal language to informal, positive to negative, or modern to 
Shakespearean. This modern implementation uses state-of-the-art transformer models with 
comprehensive evaluation metrics and a web interface.

Features:
- Multiple AI models (T5, GPT-2, BART)
- Comprehensive style categories
- Real-time evaluation metrics (BLEU, ROUGE, BERT Score, Semantic Similarity)
- Web UI with Streamlit
- Mock database for sample data
- Batch processing capabilities
- Modern configuration management

Usage:
1. Run the basic demo: python 0200.py
2. Run the full system: python style_transfer.py
3. Launch web UI: streamlit run app.py
4. Install dependencies: pip install -r requirements.txt
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from style_transfer import TextStyleTransfer
    from database import StyleTransferDatabase
    print("✅ Modern style transfer system loaded successfully!")
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Falling back to basic implementation...")
    
    # Fallback to basic implementation
    from transformers import pipeline
    
    def basic_demo():
        """Basic text style transfer demo."""
        print("🧠 Basic Text Style Transfer Demo\n")
        
        # Load a simple model
        try:
            style_transfer = pipeline("text2text-generation", model="prithivida/parrot_paraphraser_on_T5")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please install transformers: pip install transformers torch")
            return
        
        # Sample sentences
        input_texts = [
            "The presentation was highly professional and informative.",
            "I am extremely disappointed with the product I received.",
            "This is awesome! I totally love it!",
            "Thou art most fair, my dearest love."
        ]
        
        print("🔤 Style Transfer Examples:\n")
        
        for text in input_texts:
            print(f"Original: {text}")
            try:
                output = style_transfer(f"paraphrase: {text}", max_length=50, num_return_sequences=2)
                for i, o in enumerate(output):
                    print(f"  ✍️ Variant {i+1}: {o['generated_text']}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
            print()
    
    basic_demo()
    sys.exit(0)

def main():
    """Main function demonstrating the modern text style transfer system."""
    print("🎭 Modern Text Style Transfer System")
    print("=" * 50)
    
    # Initialize systems
    print("Loading AI models and database...")
    style_transfer = TextStyleTransfer()
    db = StyleTransferDatabase()
    
    # Get available options
    available_models = style_transfer.get_available_models()
    available_styles = style_transfer.get_available_styles()
    
    print(f"✅ Loaded {len(available_models)} models: {', '.join(available_models)}")
    print(f"✅ Available styles: {', '.join(available_styles)}")
    
    # Sample texts from database
    sample_texts = db.get_sample_texts(limit=4)
    
    print("\n🔤 Style Transfer Examples:")
    print("-" * 30)
    
    for sample in sample_texts:
        text = sample['text']
        print(f"\nOriginal ({sample['style_type']}): {text}")
        
        # Try different style transfers
        for style in available_styles[:2]:  # Limit to first 2 styles
            try:
                results = style_transfer.transfer_style(text, style)
                if results:
                    print(f"\n  Style: {style}")
                    for i, result in enumerate(results[:2]):  # Show first 2 results
                        print(f"    {i+1}. {result['text']}")
                        
                        # Show evaluation metrics
                        eval_metrics = style_transfer.evaluate_transfer(text, result['text'])
                        print(f"       📊 Semantic Sim: {eval_metrics.get('semantic_similarity', 0):.3f}")
                        print(f"       📊 ROUGE-L: {eval_metrics.get('rougeL', 0):.3f}")
                        print(f"       📊 BERT F1: {eval_metrics.get('bert_f1', 0):.3f}")
                        
                        # Save to database
                        db.save_transfer_result(
                            original_text=text,
                            transferred_text=result['text'],
                            style_category=style,
                            model_name=result['model'],
                            confidence_score=result['confidence'],
                            evaluation_metrics=eval_metrics
                        )
            except Exception as e:
                print(f"    ❌ Error with style {style}: {str(e)}")
    
    print("\n🎯 Summary:")
    print("-" * 20)
    print("✅ Text style transfer completed successfully!")
    print("✅ Results saved to database")
    print("✅ Evaluation metrics calculated")
    print("\n🚀 Next steps:")
    print("  - Run 'streamlit run app.py' for the web interface")
    print("  - Check 'data/style_transfer_samples.db' for stored results")
    print("  - Modify 'config.yaml' to customize models and styles")

if __name__ == "__main__":
    main()