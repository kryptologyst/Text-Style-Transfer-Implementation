"""
Modern Text Style Transfer Implementation
Supports multiple models, evaluation metrics, and comprehensive style categories.
"""

import torch
import yaml
import logging
from typing import List, Dict, Any, Optional, Tuple
from transformers import (
    pipeline, 
    T5ForConditionalGeneration, 
    T5Tokenizer,
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer
)
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from sacrebleu import BLEU
import bert_score
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextStyleTransfer:
    """Modern text style transfer implementation with multiple models and evaluation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the style transfer system."""
        self.console = Console()
        self.config = self._load_config(config_path)
        self.models = {}
        self.evaluator = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Load models
        self._load_models()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'MODELS': {
                't5_paraphrase': {
                    'name': 'prithivida/parrot_paraphraser_on_T5',
                    'type': 'text2text-generation',
                    'max_length': 50,
                    'num_return_sequences': 3,
                    'temperature': 0.7,
                    'do_sample': True
                }
            },
            'STYLE_CATEGORIES': {
                'formal_to_informal': {
                    'description': 'Convert formal language to casual/informal',
                    'prompt_template': 'Rewrite this formal text in a casual, informal style: {text}'
                }
            }
        }
    
    def _load_models(self):
        """Load all configured models."""
        self.console.print("[bold blue]Loading models...[/bold blue]")
        
        for model_name, model_config in self.config['MODELS'].items():
            try:
                self.console.print(f"Loading {model_name}...")
                
                if model_config['type'] == 'text2text-generation':
                    self.models[model_name] = pipeline(
                        model_config['type'],
                        model=model_config['name'],
                        device=0 if torch.cuda.is_available() else -1
                    )
                elif model_config['type'] == 'text-generation':
                    self.models[model_name] = pipeline(
                        model_config['type'],
                        model=model_config['name'],
                        device=0 if torch.cuda.is_available() else -1
                    )
                
                logger.info(f"Successfully loaded {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {str(e)}")
                continue
        
        self.console.print(f"[green]Loaded {len(self.models)} models successfully[/green]")
    
    def transfer_style(self, text: str, style_category: str, 
                     model_name: str = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Transfer text style using specified model and style category.
        
        Args:
            text: Input text to transform
            style_category: Style category from config
            model_name: Model to use (defaults to first available)
            **kwargs: Additional model parameters
        
        Returns:
            List of transfer results with metadata
        """
        if not self.models:
            raise ValueError("No models loaded")
        
        if model_name is None:
            model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if style_category not in self.config['STYLE_CATEGORIES']:
            raise ValueError(f"Style category {style_category} not found")
        
        model = self.models[model_name]
        model_config = self.config['MODELS'][model_name]
        style_config = self.config['STYLE_CATEGORIES'][style_category]
        
        # Prepare input
        prompt = style_config['prompt_template'].format(text=text)
        
        # Generate transfer
        try:
            if model_config['type'] == 'text2text-generation':
                # Merge model config with kwargs
                generation_params = {**model_config, **kwargs}
                results = model(prompt, **generation_params)
                
                # Format results
                formatted_results = []
                for i, result in enumerate(results):
                    formatted_results.append({
                        'text': result['generated_text'],
                        'confidence': result.get('score', 0.0),
                        'model': model_name,
                        'style_category': style_category,
                        'original_text': text
                    })
                
                return formatted_results
            
            elif model_config['type'] == 'text-generation':
                generation_params = {**model_config, **kwargs}
                results = model(prompt, **generation_params)
                
                formatted_results = []
                for i, result in enumerate(results):
                    formatted_results.append({
                        'text': result['generated_text'],
                        'confidence': result.get('score', 0.0),
                        'model': model_name,
                        'style_category': style_category,
                        'original_text': text
                    })
                
                return formatted_results
                
        except Exception as e:
            logger.error(f"Error during style transfer: {str(e)}")
            return []
    
    def evaluate_transfer(self, original_text: str, transferred_text: str) -> Dict[str, float]:
        """
        Evaluate the quality of style transfer using multiple metrics.
        
        Args:
            original_text: Original input text
            transferred_text: Transferred output text
        
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        try:
            # Semantic Similarity (using sentence transformers)
            original_embedding = self.evaluator.encode([original_text])
            transferred_embedding = self.evaluator.encode([transferred_text])
            semantic_similarity = np.dot(original_embedding[0], transferred_embedding[0])
            metrics['semantic_similarity'] = float(semantic_similarity)
            
            # ROUGE scores
            rouge_scores = self.rouge_scorer.score(original_text, transferred_text)
            metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
            metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
            metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
            
            # BLEU score
            bleu = BLEU()
            bleu_score = bleu.corpus_score([transferred_text], [[original_text]])
            metrics['bleu'] = bleu_score.score / 100.0  # Normalize to 0-1
            
            # BERT Score
            P, R, F1 = bert_score.score([transferred_text], [original_text], lang="en")
            metrics['bert_precision'] = float(P[0])
            metrics['bert_recall'] = float(R[0])
            metrics['bert_f1'] = float(F1[0])
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            metrics['error'] = str(e)
        
        return metrics
    
    def batch_transfer(self, texts: List[str], style_category: str, 
                      model_name: str = None, **kwargs) -> List[List[Dict[str, Any]]]:
        """
        Perform style transfer on multiple texts.
        
        Args:
            texts: List of input texts
            style_category: Style category to apply
            model_name: Model to use
            **kwargs: Additional parameters
        
        Returns:
            List of transfer results for each input text
        """
        results = []
        
        for text in track(texts, description="Processing texts..."):
            transfer_results = self.transfer_style(text, style_category, model_name, **kwargs)
            results.append(transfer_results)
        
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_available_styles(self) -> List[str]:
        """Get list of available style categories."""
        return list(self.config['STYLE_CATEGORIES'].keys())
    
    def display_results(self, results: List[Dict[str, Any]], show_evaluation: bool = True):
        """Display transfer results in a formatted table."""
        if not results:
            self.console.print("[red]No results to display[/red]")
            return
        
        table = Table(title="Style Transfer Results")
        table.add_column("Original Text", style="cyan", max_width=30)
        table.add_column("Transferred Text", style="green", max_width=30)
        table.add_column("Model", style="blue")
        table.add_column("Style", style="magenta")
        table.add_column("Confidence", style="yellow")
        
        if show_evaluation:
            table.add_column("Semantic Sim.", style="yellow")
            table.add_column("ROUGE-L", style="yellow")
            table.add_column("BERT F1", style="yellow")
        
        for result in results:
            row = [
                result['original_text'][:50] + "..." if len(result['original_text']) > 50 else result['original_text'],
                result['text'][:50] + "..." if len(result['text']) > 50 else result['text'],
                result['model'],
                result['style_category'],
                f"{result['confidence']:.3f}" if result['confidence'] else "N/A"
            ]
            
            if show_evaluation:
                eval_metrics = self.evaluate_transfer(result['original_text'], result['text'])
                row.extend([
                    f"{eval_metrics.get('semantic_similarity', 0):.3f}",
                    f"{eval_metrics.get('rougeL', 0):.3f}",
                    f"{eval_metrics.get('bert_f1', 0):.3f}"
                ])
            
            table.add_row(*row)
        
        self.console.print(table)

def main():
    """Main function to demonstrate the style transfer system."""
    # Initialize the system
    style_transfer = TextStyleTransfer()
    
    # Sample texts
    sample_texts = [
        "The presentation was highly professional and informative.",
        "I am extremely disappointed with the product I received.",
        "This is awesome! I totally love it!",
        "Thou art most fair, my dearest love."
    ]
    
    # Available styles
    available_styles = style_transfer.get_available_styles()
    available_models = style_transfer.get_available_models()
    
    print(f"Available models: {available_models}")
    print(f"Available styles: {available_styles}")
    
    # Perform transfers
    for text in sample_texts:
        print(f"\n{'='*60}")
        print(f"Original: {text}")
        
        # Try different style transfers
        for style in available_styles[:2]:  # Limit to first 2 styles for demo
            try:
                results = style_transfer.transfer_style(text, style)
                if results:
                    print(f"\nStyle: {style}")
                    for i, result in enumerate(results[:2]):  # Show first 2 results
                        print(f"  {i+1}. {result['text']}")
                        
                        # Show evaluation metrics
                        eval_metrics = style_transfer.evaluate_transfer(text, result['text'])
                        print(f"     Semantic Similarity: {eval_metrics.get('semantic_similarity', 0):.3f}")
                        print(f"     ROUGE-L: {eval_metrics.get('rougeL', 0):.3f}")
                        print(f"     BERT F1: {eval_metrics.get('bert_f1', 0):.3f}")
            except Exception as e:
                print(f"Error with style {style}: {str(e)}")

if __name__ == "__main__":
    main()
