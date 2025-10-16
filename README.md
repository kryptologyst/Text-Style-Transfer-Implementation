# Text Style Transfer Implementation

A comprehensive text style transfer system that transforms text while preserving meaning using state-of-the-art AI models.

## Features

- **Multiple AI Models**: Support for T5, GPT-2, BART, and other transformer models
- **Comprehensive Style Categories**: Formal/informal, positive/negative, modern/Shakespearean, and more
- **Real-time Evaluation**: BLEU, ROUGE, BERT Score, and semantic similarity metrics
- **Modern Web UI**: Interactive Streamlit interface with visualizations
- **Mock Database**: SQLite database with sample texts and transfer history
- **Batch Processing**: Process multiple texts simultaneously
- **Configuration Management**: YAML-based configuration system
- **Performance Monitoring**: Logging and performance tracking

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Text-Style-Transfer-Implementation.git
cd Text-Style-Transfer-Implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the basic demo:
```bash
python 0200.py
```

4. Launch the web interface:
```bash
streamlit run app.py
```

### Basic Usage

```python
from style_transfer import TextStyleTransfer

# Initialize the system
style_transfer = TextStyleTransfer()

# Transfer text style
results = style_transfer.transfer_style(
    text="The presentation was highly professional and informative.",
    style_category="formal_to_informal",
    model_name="t5_paraphrase"
)

# Display results
for result in results:
    print(f"Transferred: {result['text']}")
    print(f"Confidence: {result['confidence']}")
```

## 📁 Project Structure

```
text-style-transfer/
├── 0200.py                 # Main demo script
├── style_transfer.py       # Core style transfer implementation
├── database.py            # Mock database implementation
├── app.py                 # Streamlit web interface
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Database and logs
│   ├── style_transfer_samples.db
│   └── logs/
└── .gitignore           # Git ignore file
```

## Style Categories

The system supports various style transfer categories:

- **Formal ↔ Informal**: Convert between professional and casual language
- **Positive ↔ Negative**: Change sentiment while preserving meaning
- **Modern ↔ Shakespearean**: Transform between contemporary and classical English
- **Custom Styles**: Easily add new style categories via configuration

## Supported Models

- **T5 Paraphrase**: `prithivida/parrot_paraphraser_on_T5`
- **GPT-2**: `gpt2` (for text generation)
- **BART**: `facebook/bart-large-cnn` (for summarization-based transfer)
- **Custom Models**: Add your own fine-tuned models via configuration

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Semantic Similarity**: Measures meaning preservation using sentence transformers
- **ROUGE Scores**: Evaluates n-gram overlap (ROUGE-1, ROUGE-2, ROUGE-L)
- **BLEU Score**: Measures translation quality
- **BERT Score**: Contextual similarity using BERT embeddings
- **Confidence Scores**: Model-generated confidence levels

## Web Interface

The Streamlit web interface provides:

- **Interactive Text Input**: Type or select sample texts
- **Model Selection**: Choose from available AI models
- **Style Configuration**: Select style categories and parameters
- **Real-time Results**: See transfer results with evaluation metrics
- **Visualizations**: Charts and graphs for metric comparison
- **Transfer History**: View and analyze past transfers
- **Database Integration**: Save and retrieve transfer results

## Configuration

Customize the system via `config.yaml`:

```yaml
MODELS:
  t5_paraphrase:
    name: "prithivida/parrot_paraphraser_on_T5"
    type: "text2text-generation"
    max_length: 50
    temperature: 0.7

STYLE_CATEGORIES:
  formal_to_informal:
    description: "Convert formal language to casual/informal"
    prompt_template: "Rewrite this formal text in a casual, informal style: {text}"
```

## Database Schema

The mock database includes three main tables:

- **sample_texts**: Pre-loaded sample texts with categories
- **style_categories**: Available style transfer categories
- **transfer_results**: Historical transfer results with metrics

## 🔧 Advanced Usage

### Batch Processing

```python
texts = ["Text 1", "Text 2", "Text 3"]
results = style_transfer.batch_transfer(
    texts=texts,
    style_category="formal_to_informal"
)
```

### Custom Evaluation

```python
metrics = style_transfer.evaluate_transfer(
    original_text="Original text",
    transferred_text="Transferred text"
)
print(f"Semantic Similarity: {metrics['semantic_similarity']}")
```

### Database Operations

```python
from database import StyleTransferDatabase

db = StyleTransferDatabase()

# Get sample texts
samples = db.get_sample_texts(style_type="formal", limit=10)

# Save transfer results
db.save_transfer_result(
    original_text="Original",
    transferred_text="Transferred",
    style_category="formal_to_informal",
    model_name="t5_paraphrase"
)
```

## Performance

- **Model Loading**: ~10-30 seconds (depending on model size)
- **Transfer Speed**: ~1-3 seconds per text (GPU recommended)
- **Evaluation**: ~0.5-1 second per text
- **Memory Usage**: ~2-8GB (depending on models loaded)

## 🛠️ Development

### Adding New Models

1. Update `config.yaml` with new model configuration
2. Add model loading logic in `style_transfer.py`
3. Test with sample texts

### Adding New Style Categories

1. Add style definition to `config.yaml`
2. Update database with sample texts
3. Test transfer functionality

### Custom Evaluation Metrics

1. Implement metric calculation in `evaluate_transfer()`
2. Add to configuration if needed
3. Update UI to display new metrics

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure sufficient memory and correct model names
2. **CUDA Errors**: Install PyTorch with CUDA support or use CPU-only version
3. **Import Errors**: Install all dependencies from `requirements.txt`
4. **Database Errors**: Ensure write permissions for data directory

### Performance Optimization

- Use GPU acceleration when available
- Adjust batch sizes based on memory
- Cache models for repeated use
- Use smaller models for faster inference

## References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Text Style Transfer Survey](https://arxiv.org/abs/1908.09395)
- [BLEU Score](https://aclanthology.org/P02-1040/)
- [ROUGE Score](https://aclanthology.org/W04-1013/)
- [BERT Score](https://arxiv.org/abs/1904.09675)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


# Text-Style-Transfer-Implementation
