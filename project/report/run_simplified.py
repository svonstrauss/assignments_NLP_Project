#!/usr/bin/env python3
"""
Simplified Analysis Runner for Health Misinformation Project
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Setup plotting
try:
    plt.style.use('seaborn-v0_8')
except:
    pass  # If style not available, use default
sns.set(font_scale=1.2)

def main():
    """Run simplified analysis"""
    print("=" * 80)
    print(" SIMPLIFIED HEALTH MISINFORMATION ANALYSIS ")
    print("=" * 80)
    
    # Create output directory for visualizations
    visualizations_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Create synthetic data
    print("\nGenerating synthetic data for demonstration...")
    np.random.seed(42)
    n_samples = 1000
    labels = np.random.choice(['fake', 'real'], size=n_samples, p=[0.4, 0.6])
    
    # Generate fake and real headlines
    fake_titles = [
        "COVID CURE FOUND: Government hiding miracle treatment!",
        "Vaccines PROVEN to cause autism in new study!",
        "5G towers spreading coronavirus, whistleblower reveals!",
        "Doctors ADMIT: Hand sanitizer causes more harm than good!",
        "Leaked documents show hospitals fabricating COVID deaths"
    ]
    
    real_titles = [
        "New COVID-19 study shows declining transmission rates",
        "Vaccine trials enter phase 3 with promising results",
        "Health officials update mask guidance for public spaces",
        "Study finds social distancing effective at reducing spread",
        "New treatment shows modest improvement in severe cases"
    ]
    
    texts = []
    for label in labels:
        if label == 'fake':
            title = np.random.choice(fake_titles)
        else:
            title = np.random.choice(real_titles)
            
        # Add some variation
        text = f"{title} {np.random.randint(1, 100)}"
        texts.append(text)
    
    # Create DataFrame
    data = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    print(f"Created dataset with {len(data)} samples")
    print(f"Class distribution: {data['label'].value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )
    
    # Train a simple model
    print("\nTraining logistic regression model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create a visualization
    from sklearn.metrics import confusion_matrix
    plt.figure(figsize=(10, 6))
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    # Plot it using seaborn
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, 'confusion_matrix.png'), dpi=300)
    
    # Create vocabulary visualization
    tfidf = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['classifier']
    
    # Get feature names and coefficients
    feature_names = tfidf.get_feature_names_out()
    coefs = classifier.coef_[0]
    
    # Create a DataFrame with feature names and coefficients
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs
    })
    
    # Sort by absolute coefficient value
    coef_df['abs_coef'] = coef_df['coefficient'].abs()
    top_coefs = coef_df.sort_values('abs_coef', ascending=False).head(20)
    
    # Plot top coefficients
    plt.figure(figsize=(12, 8))
    colors = ['red' if x < 0 else 'blue' for x in top_coefs['coefficient']]
    
    plt.barh(top_coefs['feature'], top_coefs['coefficient'], color=colors)
    plt.title('Top Features for Fake News Classification', fontsize=16)
    plt.xlabel('Coefficient (Red = Contributes to Real, Blue = Contributes to Fake)', fontsize=12)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, 'top_features.png'), dpi=300)
    
    print(f"\nAnalysis complete! Visualizations saved to {visualizations_dir}")

if __name__ == "__main__":
    main()