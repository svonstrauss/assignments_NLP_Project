#!/usr/bin/env python3
"""
Enhanced Analysis Runner for Health Misinformation Project
This script runs all enhanced models and generates visualizations
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import our enhanced modules
from enhanced_models import HealthMisinformationDetector
from cross_domain_transfer import CrossDomainTransfer
# Importing advanced_transformers and checking if transformers are available
try:
    from advanced_transformers import TransformerModels, TRANSFORMERS_AVAILABLE
except ImportError:
    print("Warning: advanced_transformers module not available. Advanced transformer models will be skipped.")
    TRANSFORMERS_AVAILABLE = False

def main():
    """Run the enhanced analysis"""
    print("=" * 80)
    print(" ENHANCED HEALTH MISINFORMATION ANALYSIS ")
    print("=" * 80)
    
    # Create output directory for visualizations
    visualizations_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Change working directory to ensure files are saved in the right place
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Step 1: Run traditional models with enhanced visualizations
    print("\n1. Running Traditional Models with Enhanced Visualizations")
    print("-" * 60)
    
    detector = HealthMisinformationDetector()
    detector.load_data(use_synthetic=True, n_samples=1000)
    detector.prepare_data()
    detector.train_traditional_models()
    detector.visualize_results()
    
    # Step 2: Cross-domain transfer analysis
    print("\n2. Running Cross-Domain Transfer Analysis")
    print("-" * 60)
    
    transfer = CrossDomainTransfer()
    covid_train, covid_test, general_test = transfer.prepare_data()
    
    # Simulate model evaluation for demonstration
    models = [
        "bert-base-uncased", 
        "roberta-base", 
        "distilbert-base-uncased",
        "domain-adapted-bert"
    ]
    
    for model_name in models:
        transfer.models[model_name] = None  # Placeholder
        transfer.evaluate_cross_domain(model_name, covid_test, general_test)
    
    transfer.visualize_domain_transfer()
    
    # Step 3: Run transformer models if available
    if TRANSFORMERS_AVAILABLE:
        print("\n3. Running Transformer Models (Demo Mode)")
        print("-" * 60)
        
        # Create transformer models object
        transformer = TransformerModels()
        
        # Use a subset of the data for demonstration
        X_train = covid_train['content'].tolist()[:50]
        y_train = covid_train['label'].tolist()[:50]
        X_test = covid_test['content'].tolist()[:20]
        y_test = covid_test['label'].tolist()[:20]
        
        # Run zero-shot learning demo
        transformer.zero_shot_learning(X_test[:10])
        
        # Run few-shot learning demo
        transformer.few_shot_learning(X_train, y_train, X_test[:10])
        
        # Note: Full BERT training would be done here but requires more resources
        print("Note: Full transformer model training skipped in demo mode (requires GPU)")
    
    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE! ")
    print("=" * 80)
    print("\nAll visualizations have been saved to the project/visualizations directory")

if __name__ == "__main__":
    main()