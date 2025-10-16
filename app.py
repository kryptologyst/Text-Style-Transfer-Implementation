"""
Streamlit Web UI for Text Style Transfer
Modern, interactive interface for text style transfer with real-time evaluation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from style_transfer import TextStyleTransfer
from database import StyleTransferDatabase
import time

# Page configuration
st.set_page_config(
    page_title="Text Style Transfer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_style_transfer():
    """Load the style transfer system (cached)."""
    return TextStyleTransfer()

@st.cache_resource
def load_database():
    """Load the database (cached)."""
    return StyleTransferDatabase()

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Text Style Transfer</h1>', unsafe_allow_html=True)
    st.markdown("Transform text while preserving meaning using state-of-the-art AI models")
    
    # Initialize systems
    with st.spinner("Loading AI models..."):
        style_transfer = load_style_transfer()
        db = load_database()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = style_transfer.get_available_models()
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            help="Choose the AI model for style transfer"
        )
        
        # Style selection
        available_styles = style_transfer.get_available_styles()
        selected_style = st.selectbox(
            "Select Style Category",
            available_styles,
            help="Choose the style transformation to apply"
        )
        
        # Advanced parameters
        st.subheader("🔧 Advanced Parameters")
        
        max_length = st.slider("Max Length", 20, 200, 50)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        num_sequences = st.slider("Number of Variants", 1, 5, 3)
        
        # Sample text selection
        st.subheader("📝 Sample Texts")
        sample_texts = db.get_sample_texts(limit=10)
        sample_options = {f"{text['text'][:50]}...": text['text'] for text in sample_texts}
        sample_options["Custom"] = "custom"
        
        selected_sample = st.selectbox("Choose Sample Text", list(sample_options.keys()))
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("✍️ Text Input")
        
        # Text input
        if selected_sample == "Custom":
            input_text = st.text_area(
                "Enter your text",
                height=150,
                placeholder="Type your text here...",
                help="Enter the text you want to transform"
            )
        else:
            input_text = sample_options[selected_sample]
            st.text_area("Selected Sample Text", value=input_text, height=150, disabled=True)
        
        # Transfer button
        if st.button("🚀 Transfer Style", type="primary", use_container_width=True):
            if input_text.strip():
                with st.spinner("Processing..."):
                    start_time = time.time()
                    
                    # Perform style transfer
                    results = style_transfer.transfer_style(
                        input_text,
                        selected_style,
                        selected_model,
                        max_length=max_length,
                        temperature=temperature,
                        num_return_sequences=num_sequences
                    )
                    
                    processing_time = time.time() - start_time
                    
                    if results:
                        # Store results in session state
                        st.session_state['transfer_results'] = results
                        st.session_state['original_text'] = input_text
                        st.session_state['processing_time'] = processing_time
                        
                        st.success(f"✅ Style transfer completed in {processing_time:.2f} seconds!")
                    else:
                        st.error("❌ Style transfer failed. Please try again.")
            else:
                st.warning("⚠️ Please enter some text to transform.")
    
    with col2:
        st.header("📊 Quick Stats")
        
        # Display model info
        st.metric("Available Models", len(available_models))
        st.metric("Style Categories", len(available_styles))
        
        if 'processing_time' in st.session_state:
            st.metric("Last Processing Time", f"{st.session_state['processing_time']:.2f}s")
    
    # Results section
    if 'transfer_results' in st.session_state:
        st.header("🎯 Transfer Results")
        
        results = st.session_state['transfer_results']
        original_text = st.session_state['original_text']
        
        # Display results
        for i, result in enumerate(results):
            with st.expander(f"Variant {i+1} - Confidence: {result['confidence']:.3f}", expanded=i==0):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Transferred Text:**")
                    st.write(result['text'])
                
                with col2:
                    # Evaluation metrics
                    eval_metrics = style_transfer.evaluate_transfer(original_text, result['text'])
                    
                    st.metric("Semantic Similarity", f"{eval_metrics.get('semantic_similarity', 0):.3f}")
                    st.metric("ROUGE-L", f"{eval_metrics.get('rougeL', 0):.3f}")
                    st.metric("BERT F1", f"{eval_metrics.get('bert_f1', 0):.3f}")
        
        # Visualization
        st.subheader("📈 Evaluation Metrics")
        
        # Prepare data for visualization
        metrics_data = []
        for i, result in enumerate(results):
            eval_metrics = style_transfer.evaluate_transfer(original_text, result['text'])
            metrics_data.append({
                'Variant': f'Variant {i+1}',
                'Semantic Similarity': eval_metrics.get('semantic_similarity', 0),
                'ROUGE-L': eval_metrics.get('rougeL', 0),
                'BERT F1': eval_metrics.get('bert_f1', 0),
                'Confidence': result['confidence']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Create radar chart
        fig = go.Figure()
        
        metrics = ['Semantic Similarity', 'ROUGE-L', 'BERT F1', 'Confidence']
        
        for i, row in df_metrics.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[metric] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=row['Variant'],
                line=dict(color=px.colors.qualitative.Set1[i])
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Evaluation Metrics Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Save results option
        if st.button("💾 Save Results to Database"):
            for result in results:
                eval_metrics = style_transfer.evaluate_transfer(original_text, result['text'])
                db.save_transfer_result(
                    original_text=original_text,
                    transferred_text=result['text'],
                    style_category=selected_style,
                    model_name=selected_model,
                    confidence_score=result['confidence'],
                    evaluation_metrics=eval_metrics
                )
            st.success("✅ Results saved to database!")
    
    # History section
    st.header("📚 Transfer History")
    
    history = db.get_transfer_history(limit=20)
    
    if history:
        # Convert to DataFrame for display
        history_df = pd.DataFrame(history)
        history_df['created_at'] = pd.to_datetime(history_df['created_at'])
        
        # Display recent transfers
        st.subheader("Recent Transfers")
        
        for _, row in history_df.head(5).iterrows():
            with st.expander(f"{row['created_at'].strftime('%Y-%m-%d %H:%M')} - {row['style_category']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original:**")
                    st.write(row['original_text'])
                
                with col2:
                    st.write("**Transferred:**")
                    st.write(row['transferred_text'])
                
                st.write(f"**Model:** {row['model_name']} | **Confidence:** {row['confidence_score']:.3f}")
        
        # Statistics
        st.subheader("📊 Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Transfers", len(history_df))
        
        with col2:
            avg_confidence = history_df['confidence_score'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        with col3:
            most_used_model = history_df['model_name'].mode().iloc[0] if not history_df.empty else "N/A"
            st.metric("Most Used Model", most_used_model)
        
        # Style category distribution
        style_counts = history_df['style_category'].value_counts()
        fig_pie = px.pie(
            values=style_counts.values,
            names=style_counts.index,
            title="Style Category Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    else:
        st.info("No transfer history available yet. Start by transferring some text!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, Transformers, and modern NLP techniques. "
        "Powered by Hugging Face models and advanced evaluation metrics."
    )

if __name__ == "__main__":
    main()
