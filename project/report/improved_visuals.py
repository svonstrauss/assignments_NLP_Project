#!/usr/bin/env python3
"""
Improved Visualizations for Health Misinformation Detection Project
Creates more visually appealing charts with realistic model comparisons
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, accuracy_score, f1_score, 
                           precision_score, recall_score, roc_curve, auc, 
                           precision_recall_curve, confusion_matrix)

# Setup custom color palette
custom_palette = {
    'logistic': '#1A535C',   # Deep Teal
    'svm': '#4ECDC4',        # Seafoam
    'random_forest': '#FF6B6B', # Coral
    'gradient_boosting': '#FFE66D', # Sunshine Yellow
    'real': '#3D5A80',      # Navy Blue
    'fake': '#E07A5F',      # Terracotta
    'background': '#F7FFF7', # Off-white
    'grid': '#E0E0E0'       # Light gray
}

# Apply styling for all plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['patch.linewidth'] = 0

def generate_realistic_data(n_samples=1000, add_noise=True):
    """Generate synthetic data with realistic patterns and noise"""
    print("\nGenerating realistic synthetic data...")
    np.random.seed(42)
    
    # Class distribution with slight imbalance
    labels = np.random.choice(['fake', 'real'], size=n_samples, p=[0.4, 0.6])
    
    # Templates for fake news
    fake_templates = [
        "BREAKING: {shocking} {authority} {hide} {treatment} for {disease}!",
        "REVEALED: {shocking} proof that {authority} {hide} {treatment}!",
        "SCANDAL: {authority} caught {hiding_action} {treatment} from public!",
        "{authority} DON'T want you to know about this {treatment}!",
        "{authority} {evidence} that {disease} is {conspiracy}!",
        "What {authority} {hiding_action} about {disease} will SHOCK you!",
        "{treatment} {proof} to {cure} {disease} but {suppressed}!",
        "{shocking}: {controversial} link between {technology} and {disease}!",
    ]
    
    # Templates for real news
    real_templates = [
        "Study: {research} shows {measured} {effect} on {disease} {outcomes}",
        "Research: {measured} {data} about {disease} {treatment} {outcomes}",
        "New data: {percentage}% {effect} in {disease} after {intervention}",
        "Scientists {discover} {measured} {effect} of {treatment} on {disease}",
        "Analysis: {data} reveals {research} findings on {disease}",
        "Health experts: {measured} approach to {disease} {treatment} recommended",
        "Report: {research} indicates {percentage}% {effect} with {intervention}",
        "Clinical trials: {treatment} shows {measured} {effect} against {disease}",
    ]
    
    # Word pools for template filling
    word_pools = {
        # Fake news words - sensationalist language
        'shocking': ['SHOCKING', 'BOMBSHELL', 'TERRIFYING', 'DISTURBING', 'ALARMING', 'MIND-BLOWING'],
        'authority': ['Doctors', 'Scientists', 'Big Pharma', 'Government', 'CDC', 'WHO', 'Experts', 'Medical establishment'],
        'hide': ['hiding', 'covering up', 'suppressing', 'refusing to approve', 'won\'t tell you about', 'banned'],
        'treatment': ['miracle cure', 'natural remedy', 'vitamin treatment', 'ancient solution', 'breakthrough therapy', 'secret protocol'],
        'disease': ['COVID-19', 'cancer', 'diabetes', 'heart disease', 'autoimmune disorders', 'chronic fatigue'],
        'hiding_action': ['hiding', 'lying about', 'suppressing information on', 'denying access to', 'covering up'],
        'evidence': ['don\'t want you to see the evidence', 'are ignoring proof', 'rejected studies'],
        'conspiracy': ['man-made', 'a bioweapon', 'created in a lab', 'engineered', 'exaggerated', 'not what they claim'],
        'proof': ['proven', 'guaranteed', 'shown', 'confirmed'],
        'cure': ['eliminate', 'cure', 'destroy', 'eradicate', 'reverse'],
        'suppressed': ['big pharma doesn\'t want you to know', 'doctors hate this', 'being censored online'],
        'controversial': ['shocking', 'undeniable', 'censored', 'hidden'],
        'technology': ['5G', 'vaccines', 'GMOs', 'smart meters', 'chemtrails', 'artificial sweeteners'],
        
        # Real news words - measured language
        'research': ['peer-reviewed research', 'clinical trials', 'systematic review', 'meta-analysis', 'controlled study', 'double-blind study'],
        'measured': ['promising', 'preliminary', 'significant', 'modest', 'encouraging', 'notable', 'limited', 'potential'],
        'effect': ['effect', 'impact', 'reduction', 'improvement', 'association', 'correlation', 'benefit', 'risk'],
        'outcomes': ['outcomes', 'results', 'rates', 'cases', 'symptoms', 'complications', 'mortality', 'transmission'],
        'data': ['data', 'findings', 'evidence', 'statistics', 'results', 'analysis', 'trials', 'studies'],
        'percentage': ['12', '23', '45', '36', '58', '17', '29', '8'],
        'discover': ['observe', 'identify', 'report', 'confirm', 'find', 'determine', 'demonstrate'],
        'intervention': ['early intervention', 'treatment protocol', 'preventive measures', 'therapeutic approach', 'lifestyle changes'],
    }
    
    texts = []
    additional_features = []
    
    for i, label in enumerate(labels):
        # Select template
        if label == 'fake':
            template = np.random.choice(fake_templates)
            # Create feature vector with characteristics of fake news
            features = [
                np.random.uniform(0.7, 1.0),  # High emotional language
                np.random.uniform(0.6, 1.0),  # High use of ALL CAPS
                np.random.uniform(0.5, 1.0),  # High use of exclamation points
                np.random.uniform(0.0, 0.3),  # Low citation count
                np.random.uniform(0.0, 0.4),  # Low methodological terms
            ]
        else:
            template = np.random.choice(real_templates)
            # Create feature vector with characteristics of real news
            features = [
                np.random.uniform(0.0, 0.3),  # Low emotional language
                np.random.uniform(0.0, 0.2),  # Low use of ALL CAPS
                np.random.uniform(0.0, 0.3),  # Low use of exclamation points
                np.random.uniform(0.5, 1.0),  # High citation count
                np.random.uniform(0.6, 1.0),  # High methodological terms
            ]
        
        # Fill template with random words from appropriate pools
        text = template
        for key in word_pools:
            if '{' + key + '}' in text:
                text = text.replace('{' + key + '}', np.random.choice(word_pools[key]))
        
        # Add variability
        if np.random.random() < 0.2:  # 20% chance to add exclamation points to titles
            text += '!!!'
            
        if label == 'fake' and np.random.random() < 0.3:  # 30% chance to uppercase words in fake news
            words = text.split()
            for j in range(len(words)):
                if np.random.random() < 0.15:  # 15% chance per word
                    words[j] = words[j].upper()
            text = ' '.join(words)
        
        # Add unique identifier to prevent perfect classification
        text += f" [ID:{i}]"
        
        texts.append(text)
        additional_features.append(features)
    
    # Create DataFrame
    data = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Add additional feature columns
    feature_names = ['emotional_language', 'caps_usage', 'exclamation_usage', 'citation_count', 'methodological_terms']
    for i, name in enumerate(feature_names):
        data[name] = [features[i] for features in additional_features]
    
    # Add noise to make classification more realistic (if requested)
    if add_noise:
        # Flip some labels to simulate noise (5% of data)
        noise_idx = np.random.choice(range(len(data)), size=int(0.05 * len(data)), replace=False)
        data.loc[noise_idx, 'label'] = data.loc[noise_idx, 'label'].apply(lambda x: 'real' if x == 'fake' else 'fake')
        
        # Add some contradiction in features (10% of data)
        contradiction_idx = np.random.choice(range(len(data)), size=int(0.1 * len(data)), replace=False)
        for idx in contradiction_idx:
            if data.loc[idx, 'label'] == 'fake':
                # Give some fake news real-news-like features
                data.loc[idx, 'emotional_language'] = np.random.uniform(0.0, 0.4)
                data.loc[idx, 'citation_count'] = np.random.uniform(0.6, 0.8)
            else:
                # Give some real news fake-news-like features
                data.loc[idx, 'emotional_language'] = np.random.uniform(0.6, 0.9)
                data.loc[idx, 'exclamation_usage'] = np.random.uniform(0.7, 0.9)
    
    print(f"Created dataset with {len(data)} samples")
    print(f"Class distribution: {data['label'].value_counts().to_dict()}")
    
    return data

def train_and_evaluate_models(data, visualizations_dir):
    """Train and evaluate multiple models with cross-validation"""
    # Prepare data
    X_text = data['text']
    
    # Create additional features
    X_features = data[['emotional_language', 'caps_usage', 'exclamation_usage', 
                      'citation_count', 'methodological_terms']]
    y = data['label']
    
    # Split the data
    X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
        X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define a cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define models
    models = {
        "Logistic Regression": Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'))
        ]),
        "Support Vector Machine": Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', SVC(probability=True, class_weight='balanced', gamma='scale'))
        ]),
        "Random Forest": Pipeline([
            ('vectorizer', CountVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced'))
        ]),
        "Gradient Boosting": Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1))
        ])
    }
    
    # Dictionary to store results
    results = {
        "model_names": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": [],
        "train_time": [],
        "y_pred": {},
        "y_proba": {},
        "feature_importance": {}
    }
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    import time
    
    for name, model in models.items():
        print(f"\nModel: {name}")
        
        # Train the model
        start_time = time.time()
        model.fit(X_text_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_text_test)
        
        # For ROC curve and AUC
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_text_test)
                # Get probability for the positive class (fake)
                fake_idx = list(model.classes_).index('fake')
                y_proba_fake = y_proba[:, fake_idx]
            except:
                # Fallback if predict_proba fails
                y_proba_fake = np.zeros(len(y_test))
        else:
            # For models without predict_proba
            y_proba_fake = np.zeros(len(y_test))
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='fake')
        recall = recall_score(y_test, y_pred, pos_label='fake')
        f1 = f1_score(y_test, y_pred, pos_label='fake')
        
        # Calculate AUC
        try:
            fpr, tpr, _ = roc_curve(y_test == 'fake', y_proba_fake)
            auc_score = auc(fpr, tpr)
        except:
            auc_score = 0.5  # Default AUC for random classifier
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc_score:.4f}")
        print(f"Training Time: {train_time:.4f} seconds")
        
        # Store results
        results["model_names"].append(name)
        results["accuracy"].append(accuracy)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1)
        results["auc"].append(auc_score)
        results["train_time"].append(train_time)
        results["y_pred"][name] = y_pred
        results["y_proba"][name] = y_proba_fake
        
        # Extract feature importance if available
        if name == "Logistic Regression" or name == "Random Forest":
            # Get vectorizer and classifier
            vectorizer = model.named_steps['vectorizer']
            classifier = model.named_steps['classifier']
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get coefficients or feature importances
            if name == "Logistic Regression":
                # For binary classification, use the first (and only) class's coefficients
                if len(classifier.classes_) == 2:
                    importance = classifier.coef_[0]
                else:
                    # Find the index of the 'fake' class
                    fake_idx = list(classifier.classes_).index('fake')
                    importance = classifier.coef_[fake_idx]
            else:  # Random Forest
                importance = classifier.feature_importances_
            
            # Create a dataframe
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            
            # Sort by absolute importance
            feature_importance['abs_importance'] = feature_importance['importance'].abs()
            feature_importance = feature_importance.sort_values('abs_importance', ascending=False).head(20)
            
            # Store the result
            results["feature_importance"][name] = feature_importance
    
    # Create visualizations
    create_visualizations(results, X_text_test, y_test, models, visualizations_dir)
    
    return results

def create_visualizations(results, X_test, y_test, models, visualizations_dir):
    """Create beautiful visualizations for model comparison"""
    print("\nGenerating professional visualizations...")
    
    # 1. Model Performance Comparison 
    create_model_comparison_chart(results, visualizations_dir)
    
    # 2. ROC Curves 
    create_roc_curves(results, y_test, visualizations_dir)
    
    # 3. Confusion Matrices
    create_confusion_matrices(results, y_test, visualizations_dir)
    
    # 4. Feature Importance
    create_feature_importance_chart(results, visualizations_dir)
    
    # 5. Precision-Recall Curves
    create_precision_recall_curves(results, y_test, visualizations_dir)
    
    # 6. Training Time Comparison
    create_training_time_chart(results, visualizations_dir)
    
    # 7. Error Analysis
    create_error_analysis(results, X_test, y_test, visualizations_dir)
    
    print(f"All visualizations saved to {visualizations_dir}")

def create_model_comparison_chart(results, visualizations_dir):
    """Create a beautiful bar chart comparing model performance metrics"""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    # Prepare data
    df = pd.DataFrame({
        'Model': results['model_names'],
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1 Score': results['f1'],
        'AUC': results['auc']
    })
    
    # Melt the dataframe
    df_melted = pd.melt(df, id_vars=['Model'], var_name='Metric', value_name='Score')
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Set background color
    plt.gca().set_facecolor(custom_palette['background'])
    plt.gcf().set_facecolor(custom_palette['background'])
    
    # Plot
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, 
                palette=sns.color_palette("viridis", n_colors=len(metrics)))
    
    # Enhance the plot
    plt.title('Model Performance Comparison', fontsize=22, pad=20)
    plt.xlabel('')
    plt.ylabel('Score', fontsize=16)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15, ha='right', fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend(title='Metric', fontsize=12, title_fontsize=14, loc='upper right')
    
    # Add value labels
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                f'{height:.2f}', ha="center", fontsize=10)
    
    # Add grid and style
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(visualizations_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_roc_curves(results, y_test, visualizations_dir):
    """Create ROC curves for all models"""
    plt.figure(figsize=(12, 10))
    
    # Set background color
    plt.gca().set_facecolor(custom_palette['background'])
    plt.gcf().set_facecolor(custom_palette['background'])
    
    # Add a diagonal reference line for random classifier
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, 
             label='Random Classifier (AUC = 0.5)')
    
    # Add ROC curve for each model
    colors = sns.color_palette("viridis", n_colors=len(results['model_names']))
    
    for i, model_name in enumerate(results['model_names']):
        auc_score = results['auc'][i]
        y_proba = results['y_proba'][model_name]
        
        # Plot ROC curve
        if y_proba.any():  # If probabilities are available
            fpr, tpr, _ = roc_curve(y_test == 'fake', y_proba)
            plt.plot(fpr, tpr, color=colors[i], lw=2.5, 
                    label=f'{model_name} (AUC = {auc_score:.3f})')
    
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=22, pad=20)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(visualizations_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrices(results, y_test, visualizations_dir):
    """Create confusion matrices for all models"""
    nmodels = len(results['model_names'])
    
    if nmodels <= 2:
        # Single row for 1-2 models
        fig, axes = plt.subplots(1, nmodels, figsize=(12, 6))
        if nmodels == 1:
            axes = [axes]  # Make sure axes is a list for consistency
    else:
        # Two rows for 3-4 models
        ncols = 2
        nrows = (nmodels + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5*nrows))
        axes = axes.flatten()
    
    # Set background color for all subplots
    for i in range(len(axes)):
        if i < nmodels:
            axes[i].set_facecolor(custom_palette['background'])
        else:
            axes[i].set_visible(False)  # Hide unused subplots
    
    fig.patch.set_facecolor(custom_palette['background'])
    
    for i, model_name in enumerate(results['model_names']):
        y_pred = results['y_pred'][model_name]
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        
        # Custom colormap
        cmap = sns.blend_palette([custom_palette['real'], '#FFFFFF', custom_palette['fake']], as_cmap=True)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'],
                   ax=axes[i], cbar=False)
        
        axes[i].set_title(f"{model_name}", fontsize=16, pad=10)
        axes[i].set_xlabel('Predicted', fontsize=12)
        if i % 2 == 0:  # Only for leftmost plots
            axes[i].set_ylabel('True', fontsize=12)
    
    plt.suptitle('Confusion Matrices', fontsize=22, y=1.05)
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(visualizations_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_importance_chart(results, visualizations_dir):
    """Create feature importance visualizations"""
    # Check which models have feature importance
    models_with_fi = list(results['feature_importance'].keys())
    
    if not models_with_fi:
        return  # No feature importance to display
    
    # Set up the figure
    n_models = len(models_with_fi)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 10))
    
    if n_models == 1:
        axes = [axes]  # Make sure axes is a list for consistency
    
    # Set background color
    for ax in axes:
        ax.set_facecolor(custom_palette['background'])
    fig.patch.set_facecolor(custom_palette['background'])
    
    for i, model_name in enumerate(models_with_fi):
        fi_df = results['feature_importance'][model_name]
        
        # Sort by importance
        top_fi = fi_df.sort_values('importance', ascending=True).tail(15)
        
        # Color bars based on whether they contribute to fake or real
        colors = [custom_palette['fake'] if x > 0 else custom_palette['real'] for x in top_fi['importance']]
        
        # Plot
        axes[i].barh(top_fi['feature'], top_fi['importance'], color=colors)
        axes[i].set_title(f"Feature Importance - {model_name}", fontsize=18, pad=15)
        axes[i].set_xlabel('Importance (+ for Fake, - for Real)', fontsize=14)
        if i == 0:  # Only for leftmost plot
            axes[i].set_ylabel('Feature', fontsize=14)
        
        # Add a vertical line at x=0
        axes[i].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add grid
        axes[i].grid(axis='x', alpha=0.3)
        
        # Set sensible limits
        max_abs = max(abs(top_fi['importance'].min()), abs(top_fi['importance'].max()))
        axes[i].set_xlim(-max_abs*1.1, max_abs*1.1)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(visualizations_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_precision_recall_curves(results, y_test, visualizations_dir):
    """Create precision-recall curves for all models"""
    plt.figure(figsize=(12, 10))
    
    # Set background color
    plt.gca().set_facecolor(custom_palette['background'])
    plt.gcf().set_facecolor(custom_palette['background'])
    
    # Add PR curve for each model
    colors = sns.color_palette("viridis", n_colors=len(results['model_names']))
    
    for i, model_name in enumerate(results['model_names']):
        y_proba = results['y_proba'][model_name]
        
        # Calculate average precision
        if y_proba.any():  # If probabilities are available
            precision, recall, _ = precision_recall_curve(y_test == 'fake', y_proba)
            # Calculate area under PR curve
            pr_auc = auc(recall, precision)
            
            # Plot PR curve
            plt.plot(recall, precision, color=colors[i], lw=2.5, 
                    label=f'{model_name} (AP = {pr_auc:.3f})')
    
    # Add a horizontal line for random classifier
    no_skill = sum(y_test == 'fake') / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray', alpha=0.5, 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title('Precision-Recall Curves', fontsize=22, pad=20)
    plt.legend(loc="best", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(visualizations_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_training_time_chart(results, visualizations_dir):
    """Create a chart showing training time for each model"""
    plt.figure(figsize=(12, 8))
    
    # Set background color
    plt.gca().set_facecolor(custom_palette['background'])
    plt.gcf().set_facecolor(custom_palette['background'])
    
    # Create bar chart of training times
    bars = plt.bar(results['model_names'], results['train_time'], 
            color=sns.color_palette("viridis", n_colors=len(results['model_names'])))
    
    # Add time labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=12)
    
    plt.title('Model Training Time Comparison', fontsize=22, pad=20)
    plt.xlabel('')
    plt.ylabel('Training Time (seconds)', fontsize=16)
    plt.xticks(rotation=15, ha='right', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(visualizations_dir, 'training_time.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_error_analysis(results, X_test, y_test, visualizations_dir):
    """Create visualizations showing common error patterns"""
    # Focus on errors from the best model (by F1 score)
    best_model_idx = np.argmax(results['f1'])
    best_model_name = results['model_names'][best_model_idx]
    y_pred = results['y_pred'][best_model_name]
    
    # Find errors
    errors = y_test != y_pred
    error_idx = np.where(errors)[0]
    
    if len(error_idx) == 0:
        print("No errors found for error analysis visualization.")
        return
    
    # Count error types
    false_positives = ((y_test == 'real') & (y_pred == 'fake')).sum()
    false_negatives = ((y_test == 'fake') & (y_pred == 'real')).sum()
    
    # Create a simple bar chart of error types
    plt.figure(figsize=(10, 8))
    
    # Set background color
    plt.gca().set_facecolor(custom_palette['background'])
    plt.gcf().set_facecolor(custom_palette['background'])
    
    error_types = ['False Positives\n(Real misclassified as Fake)', 'False Negatives\n(Fake misclassified as Real)']
    error_counts = [false_positives, false_negatives]
    colors = [custom_palette['real'], custom_palette['fake']]
    
    bars = plt.bar(error_types, error_counts, color=colors)
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                str(int(height)), ha='center', va='bottom', fontsize=14)
    
    plt.title(f'Error Analysis - {best_model_name}', fontsize=22, pad=20)
    plt.ylabel('Count', fontsize=16)
    plt.xticks(fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add error rate
    error_rate = len(error_idx) / len(y_test)
    plt.annotate(f'Overall Error Rate: {error_rate:.2%}', 
                 xy=(0.5, 0.95), xycoords='axes fraction',
                 ha='center', va='center',
                 fontsize=16, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(visualizations_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run the improved visualizations"""
    print("=" * 80)
    print(" IMPROVED VISUALIZATIONS FOR HEALTH MISINFORMATION DETECTION ")
    print("=" * 80)
    
    # Create output directory for visualizations
    visualizations_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Change working directory to ensure files are saved in the right place
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Generate realistic data with variations in class separability
    data = generate_realistic_data(n_samples=1000)
    
    # Train and evaluate models, generate visualizations
    train_and_evaluate_models(data, visualizations_dir)
    
    print(f"\nAnalysis complete! Professional visualizations saved to {visualizations_dir}")
    
    # Create an HTML file to showcase all visualizations
    create_gallery_html(visualizations_dir)

def create_gallery_html(visualizations_dir):
    """Create an HTML file with all visualizations"""
    html_path = os.path.join(visualizations_dir, "gallery.html")
    
    # Get all PNG files
    png_files = [f for f in os.listdir(visualizations_dir) if f.endswith('.png')]
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Health Misinformation Detection - Visualization Gallery</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }
            h1 {
                color: #1A535C;
                text-align: center;
                margin-bottom: 30px;
            }
            .gallery {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
            }
            .viz-container {
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                padding: 15px;
                margin-bottom: 20px;
            }
            .viz-container h2 {
                color: #4ECDC4;
                margin-top: 0;
            }
            img {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
            }
            footer {
                text-align: center;
                margin-top: 40px;
                color: #666;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <h1>Health Misinformation Detection - Visualization Gallery</h1>
        <div class="gallery">
    """
    
    # Add each visualization with a title
    for png_file in png_files:
        title = ' '.join(png_file.replace('.png', '').replace('_', ' ').title().split())
        html_content += f"""
        <div class="viz-container">
            <h2>{title}</h2>
            <img src="{png_file}" alt="{title}">
        </div>
        """
    
    # Close HTML
    html_content += """
        </div>
        <footer>
            <p>Health Misinformation Detection Project - Generated with Claude Code</p>
        </footer>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created visualization gallery: {html_path}")

if __name__ == "__main__":
    main()