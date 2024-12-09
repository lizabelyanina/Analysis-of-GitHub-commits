import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import json
import sys

def analyze_model(model_path="./fine_tuned_model"):
    """Analyze the computational requirements of a fine-tuned DistilBERT model"""
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get model size
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    
    # Get model architecture details
    architecture_info = {
        "model_type": model.config.model_type,
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "dim": model.config.dim,
        "max_position_embeddings": model.config.max_position_embeddings,
        "num_labels": model.config.num_labels
    }
    
    # Calculate storage requirements
    storage_size = 0
    for root, dirs, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            storage_size += os.path.getsize(file_path)
    
    storage_mb = storage_size / 1024**2
    
    # Estimate memory requirements for inference
    # Create a sample input
    sample_text = "sample commit message for testing"
    inputs = tokenizer(sample_text, return_tensors="pt")
    
    # Track memory before inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated()
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calculate memory used for inference
    if torch.cuda.is_available():
        inference_memory = (torch.cuda.memory_allocated() - memory_before) / 1024**2
    else:
        inference_memory = "N/A (CPU only)"
    
    return {
        "model_size_mb": size_mb,
        "storage_size_mb": storage_mb,
        "architecture": architecture_info,
        "inference_memory_mb": inference_memory,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

if __name__ == "__main__":
    analysis = analyze_model()
    print(json.dumps(analysis, indent=2))