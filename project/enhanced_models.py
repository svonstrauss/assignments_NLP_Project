"""
Enhanced Models for Health Misinformation Detection
Advanced models including BERT, RoBERTa, and zero/few-shot approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import json
from tqdm import tqdm
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# For beautiful visualizations
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Setup visualization
plt.style.use('seaborn-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

# Custom color palettes for visualizations
colors = {
    'fake_news': '#E63946',    # Bright red for fake news
    'real_news': '#457B9D',    # Blue for real news
    'gradient': ['#1D3557', '#457B9D', '#A8DADC', '#F1FAEE', '#E63946'],  # American Independence palette
    'charts': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']      # Sunset palette
}

class HealthMisinformationDetector:
    """Advanced health misinformation detection system"""
    
    def __init__(self, data_path=None):
        """Initialize the detector"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.model_results = {}
        
        # Ensure NLTK resources
        for resource in ['punkt', 'stopwords', 'wordnet']:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            except LookupError:
                nltk.download(resource)
    
    def load_data(self, path=None, use_synthetic=False, n_samples=1000):
        """Load real data or create synthetic data for testing"""
        if path:
            self.data_path = path
            
        try:
            if not use_synthetic and self.data_path:
                self.data = pd.read_csv(self.data_path)
                print(f"Loaded dataset with {len(self.data)} records")
            else:
                raise Exception("Using synthetic data")
        except Exception:
            print("Creating synthetic data for demonstration")
            np.random.seed(42)
            n_samples = n_samples
            labels = np.random.choice(['fake', 'real'], size=n_samples, p=[0.4, 0.6])
            
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
            
            fake_content = [
                "Anonymous doctors have revealed the government is hiding cures.",
                "A suppressed study shows clear links between vaccines and autism.",
                "The radiation from 5G towers is activating the virus particles.",
                "Chemicals in hand sanitizer enter your bloodstream and damage organs.",
                "Hospitals are being paid to classify deaths as COVID-related."
            ]
            
            real_content = [
                "Peer-reviewed research indicates transmission rates are decreasing.",
                "Double-blind clinical trials show the vaccine is safe and effective.",
                "Based on new evidence, health officials recommend masks indoors.",
                "Data analysis confirms social distancing reduces infection rates.",
                "Randomized controlled trial shows modest benefits of new treatment."
            ]
            
            # Generate more varied synthetic data
            titles = []
            contents = []
            
            for label in labels:
                if label == 'fake':
                    title_template = np.random.choice(fake_titles)
                    content_template = np.random.choice(fake_content)
                else:
                    title_template = np.random.choice(real_titles)
                    content_template = np.random.choice(real_content)
                
                # Add some variation to make each entry unique
                title = f"{title_template} {np.random.randint(1, 100)}"
                content = f"{content_template} {np.random.randint(1, 100)}"
                
                titles.append(title)
                contents.append(content)
            
            self.data = pd.DataFrame({
                'title': titles,
                'content': contents,
                'label': labels
            })
        
        # Display basic info
        print(f"\nDataset shape: {self.data.shape}")
        return self.data
    
    def preprocess_text(self, text, remove_stopwords=True):
        """Advanced text preprocessing pipeline"""
        if not isinstance(text, str): 
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace URLs, emails, mentions
        text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text)
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        text = re.sub(r'@\w+', '[USER]', text)
        
        # Replace hashtags with just the word
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Handle COVID abbreviations and common variants
        text = text.replace("covid", "covid19")
        text = text.replace("covid-19", "covid19")
        text = text.replace("coronavirus", "covid19")
        text = text.replace("corona virus", "covid19")
        
        # Health-specific preprocessing
        health_replacements = {
            "vaxx": "vaccine",
            "vaxxed": "vaccinated",
            "antivaxx": "antivaccine",
            "jab": "vaccine",
            "vax": "vaccine",
        }
        
        for original, replacement in health_replacements.items():
            text = re.sub(r'\b' + original + r'\b', replacement, text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        if remove_stopwords:
            clean_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        else:
            clean_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(clean_tokens)
    
    def prepare_data(self, text_col='content'):
        """Prepare data for modeling"""
        if self.data is None:
            self.load_data()
        
        self.model_data = self.data.copy()
        self.text_col = text_col if text_col in self.model_data.columns else 'title'
        
        print(f"Processing text from '{self.text_col}' column...")
        self.model_data['processed_text'] = self.model_data[self.text_col].apply(
            lambda x: self.preprocess_text(x)
        )
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.model_data['processed_text'], 
            self.model_data['label'], 
            test_size=0.2, 
            random_state=42,
            stratify=self.model_data['label']
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_traditional_models(self):
        """Train and evaluate traditional ML models"""
        if not hasattr(self, 'X_train'):
            self.prepare_data()
        
        model_configs = {
            'logistic_regression': {
                'name': 'Logistic Regression',
                'classifier': LogisticRegression(max_iter=1000, C=1.0),
                'color': colors['charts'][0]
            },
            'svm': {
                'name': 'Support Vector Machine',
                'classifier': SVC(probability=True),
                'color': colors['charts'][1]
            },
            'random_forest': {
                'name': 'Random Forest',
                'classifier': RandomForestClassifier(n_estimators=100),
                'color': colors['charts'][2]
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting',
                'classifier': GradientBoostingClassifier(),
                'color': colors['charts'][3]
            }
        }
        
        results = {}
        
        # TF-IDF settings
        tfidf = TfidfVectorizer(
            max_features=10000, 
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.9
        )
        
        X_train_tfidf = tfidf.fit_transform(self.X_train)
        X_test_tfidf = tfidf.transform(self.X_test)
        
        for model_key, config in model_configs.items():
            print(f"\nTraining {config['name']}...")
            classifier = config['classifier']
            
            # Train model
            classifier.fit(X_train_tfidf, self.y_train)
            
            # Predictions
            y_pred = classifier.predict(X_test_tfidf)
            y_prob = classifier.predict_proba(X_test_tfidf)[:, 1] if hasattr(classifier, 'predict_proba') else None
            
            # Get metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            
            # Save model and results
            self.models[model_key] = {
                'classifier': classifier,
                'vectorizer': tfidf,
                'config': config
            }
            
            results[model_key] = {
                'name': config['name'],
                'accuracy': accuracy,
                'report': report,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'color': config['color'],
                'feature_importance': self._get_feature_importance(classifier, tfidf) if model_key in ['logistic_regression', 'random_forest'] else None
            }
            
            print(f"{config['name']} Accuracy: {accuracy:.4f}")
        
        self.model_results.update(results)
        return results
    
    def _get_feature_importance(self, model, vectorizer):
        """Extract feature importance from models"""
        features = vectorizer.get_feature_names_out()
        
        if hasattr(model, 'coef_'):
            # For linear models like Logistic Regression
            importance = model.coef_[0]
        elif hasattr(model, 'feature_importances_'):
            # For tree-based models like Random Forest
            importance = model.feature_importances_
        else:
            return None
        
        # Create dataframe with features and their importance scores
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': importance
        })
        
        # Sort by absolute importance
        feature_importance['abs_importance'] = abs(feature_importance['importance'])
        feature_importance = feature_importance.sort_values('abs_importance', ascending=False)
        
        return feature_importance.head(30)  # Return top 30 features
    
    def visualize_results(self):
        """Create beautiful visualizations of model results"""
        if not self.model_results:
            print("No models trained yet. Please train models first.")
            return
        
        # 1. Model Performance Comparison
        self._plot_model_comparison()
        
        # 2. Confusion Matrices
        self._plot_confusion_matrices()
        
        # 3. ROC Curves
        self._plot_roc_curves()
        
        # 4. Feature Importance
        self._plot_feature_importance()
        
        # 5. Error Analysis
        self._plot_error_analysis()
        
        # 6. Word Clouds
        self._plot_word_clouds()
    
    def _plot_model_comparison(self):
        """Plot model performance comparison"""
        model_names = []
        accuracy = []
        precision = []
        recall = []
        f1 = []
        colors_list = []
        
        for model_key, results in self.model_results.items():
            model_names.append(results['name'])
            accuracy.append(results['accuracy'])
            precision.append(results['report']['fake']['precision'])
            recall.append(results['report']['fake']['recall'])
            f1.append(results['report']['fake']['f1-score'])
            colors_list.append(results['color'])
        
        # Create a DataFrame
        df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
        
        # Create a melted version for seaborn plotting
        df_melted = pd.melt(df, id_vars=['Model'], var_name='Metric', value_name='Score')
        
        # Plot with custom styling
        plt.figure(figsize=(14, 8))
        chart = sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, palette=colors['charts'][:4])
        
        # Customize
        plt.title('Model Performance Comparison', fontsize=20, fontweight='bold')
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=15)
        plt.legend(title='Metric', fontsize=12, title_fontsize=14)
        
        # Add value labels on top of bars
        for p in chart.patches:
            height = p.get_height()
            chart.text(p.get_x() + p.get_width()/2.,
                    height + 0.01,
                    f'{height:.2f}',
                    ha="center", fontsize=10)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.model_results)
        fig, axes = plt.subplots(1, n_models, figsize=(n_models*5, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_key, results) in enumerate(self.model_results.items()):
            cm = confusion_matrix(self.y_test, results['y_pred'])
            
            # Normalize the confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create a custom colormap transition from blue to white to red
            cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#457B9D', '#F1FAEE', '#E63946'])
            
            sns.heatmap(cm_norm, annot=cm, fmt='d', cmap=cmap, 
                        xticklabels=['Real', 'Fake'], 
                        yticklabels=['Real', 'Fake'],
                        ax=axes[i], cbar=False)
            
            axes[i].set_title(f"{results['name']}", fontsize=14, fontweight='bold')
            if i == 0:
                axes[i].set_ylabel('True Label', fontsize=12)
            axes[i].set_xlabel('Predicted Label', fontsize=12)
        
        plt.suptitle('Confusion Matrices', fontsize=18, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_key, results in self.model_results.items():
            if results['y_prob'] is not None:
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(
                    (self.y_test == 'fake').astype(int), 
                    results['y_prob']
                )
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, lw=2, label=f"{results['name']} (AUC = {roc_auc:.3f})", 
                         color=results['color'])
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=18, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_importance(self):
        """Plot feature importance for models that provide it"""
        # Find models with feature importance
        feature_models = {k: v for k, v in self.model_results.items() if v['feature_importance'] is not None}
        
        if not feature_models:
            return
        
        n_models = len(feature_models)
        fig, axes = plt.subplots(1, n_models, figsize=(n_models*8, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_key, results) in enumerate(feature_models.items()):
            fi = results['feature_importance']
            top_n = 15  # Show top 15 features
            
            # Sort by importance to get most important features
            fi_sorted = fi.sort_values('importance', ascending=True).tail(top_n)
            
            # Color bars based on if they're positive (contributes to 'fake') or negative (contributes to 'real')
            colors = [colors['fake_news'] if x > 0 else colors['real_news'] for x in fi_sorted['importance']]
            
            # Create barplot
            sns.barplot(x='importance', y='feature', data=fi_sorted, 
                        ax=axes[i], palette=colors)
            
            axes[i].set_title(f"Top Features - {results['name']}", fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Importance', fontsize=12)
            if i == 0:
                axes[i].set_ylabel('Feature', fontsize=12)
            else:
                axes[i].set_ylabel('')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors['fake_news'], label='Contributes to Fake News'),
                Patch(facecolor=colors['real_news'], label='Contributes to Real News')
            ]
            axes[i].legend(handles=legend_elements, loc='lower right')
            
            # Add gridlines
            axes[i].grid(axis='x', alpha=0.3)
        
        plt.suptitle('Feature Importance Analysis', fontsize=18, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_error_analysis(self):
        """Plot error analysis - examine where models fail"""
        # Get misclassified examples from the best model
        best_model_key = max(self.model_results, key=lambda k: self.model_results[k]['accuracy'])
        best_results = self.model_results[best_model_key]
        
        # Create mask for misclassified examples
        y_test_array = np.array(self.y_test)
        misclassified = y_test_array != best_results['y_pred']
        
        if sum(misclassified) == 0:
            print("No misclassified examples found.")
            return
        
        # Get the original texts and labels for misclassified examples
        misclassified_idx = np.where(misclassified)[0]
        misclassified_texts = np.array(self.X_test)[misclassified_idx]
        misclassified_true = y_test_array[misclassified_idx]
        misclassified_pred = best_results['y_pred'][misclassified_idx]
        
        # Original texts (before preprocessing)
        original_texts = self.model_data.loc[self.model_data['processed_text'].isin(misclassified_texts), self.text_col].values
        
        # Count common error transitions
        error_transitions = pd.DataFrame({
            'True': misclassified_true,
            'Predicted': misclassified_pred
        })
        
        transition_counts = error_transitions.groupby(['True', 'Predicted']).size().reset_index(name='Count')
        
        # Plot error transitions
        plt.figure(figsize=(8, 6))
        sns.barplot(x='True', y='Count', hue='Predicted', data=transition_counts, 
                    palette=[colors['fake_news'], colors['real_news']])
        
        plt.title('Error Transition Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('True Label', fontsize=14)
        plt.ylabel('Count of Errors', fontsize=14)
        plt.legend(title='Predicted As', fontsize=12, title_fontsize=14)
        
        plt.tight_layout()
        plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Error Word Cloud
        if len(original_texts) > 0:
            error_text = " ".join(original_texts)
            
            plt.figure(figsize=(10, 8))
            error_wordcloud = WordCloud(
                width=800, height=600,
                background_color='white',
                colormap=cm.copper_r,  # Copper-red color palette
                contour_width=1, contour_color='#E63946'
            ).generate(error_text)
            
            plt.imshow(error_wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Words in Misclassified Examples', fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.savefig('error_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def _plot_word_clouds(self):
        """Generate word clouds for fake and real news"""
        if 'label' not in self.model_data.columns or self.text_col not in self.model_data.columns:
            return
        
        fake_text = " ".join(self.model_data[self.model_data['label'] == 'fake'][self.text_col].astype(str))
        real_text = " ".join(self.model_data[self.model_data['label'] == 'real'][self.text_col].astype(str))
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Custom colormaps
        fake_cmap = LinearSegmentedColormap.from_list('fake_cmap', ['#E63946', '#F1FAEE'])
        real_cmap = LinearSegmentedColormap.from_list('real_cmap', ['#457B9D', '#A8DADC'])
        
        # Fake news word cloud
        fake_wordcloud = WordCloud(
            width=800, height=600,
            background_color='white',
            colormap=fake_cmap,
            contour_width=1, contour_color='#E63946'
        ).generate(fake_text)
        
        axes[0].imshow(fake_wordcloud, interpolation='bilinear')
        axes[0].axis('off')
        axes[0].set_title('Fake News Word Cloud', fontsize=18, fontweight='bold', color=colors['fake_news'])
        
        # Real news word cloud
        real_wordcloud = WordCloud(
            width=800, height=600,
            background_color='white',
            colormap=real_cmap,
            contour_width=1, contour_color='#457B9D'
        ).generate(real_text)
        
        axes[1].imshow(real_wordcloud, interpolation='bilinear')
        axes[1].axis('off')
        axes[1].set_title('Real News Word Cloud', fontsize=18, fontweight='bold', color=colors['real_news'])
        
        plt.tight_layout()
        plt.savefig('word_clouds.png', dpi=300, bbox_inches='tight')
        plt.show()

# Code to run this module
if __name__ == "__main__":
    # Initialize detector
    detector = HealthMisinformationDetector()
    
    # Load data - will use synthetic for demonstration
    detector.load_data(use_synthetic=True, n_samples=1000)
    
    # Prepare data
    detector.prepare_data()
    
    # Train traditional models
    detector.train_traditional_models()
    
    # Visualize results
    detector.visualize_results()
    
    print("\nAnalysis complete! Visualizations have been saved.")