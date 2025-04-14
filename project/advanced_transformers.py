"""
Advanced Transformer Models for Health Misinformation Detection
Includes BERT, RoBERTa, zero-shot and few-shot learning approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Try to import transformer-related libraries, but provide fallbacks
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import transformers
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
        pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM, get_scheduler
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import transformer libraries - {str(e)}")
    TRANSFORMERS_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

# Custom color palettes for visualizations
colors = {
    'fake_news': '#E63946',    # Bright red for fake news
    'real_news': '#457B9D',    # Blue for real news
    'gradient': ['#1D3557', '#457B9D', '#A8DADC', '#F1FAEE', '#E63946'],  # American Independence palette
    'charts': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']      # Sunset palette
}

# Only define NewsDataset if transformers is available
if TRANSFORMERS_AVAILABLE:
    class NewsDataset(Dataset):
        """Dataset for transformer models"""
        def __init__(self, texts, labels, tokenizer, max_length=512):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = 1 if self.labels[idx] == 'fake' else 0
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'][0],
                'attention_mask': encoding['attention_mask'][0],
                'labels': torch.tensor(label, dtype=torch.long)
            }

class TransformerModels:
    """Class for training and evaluating transformer models"""
    
    def __init__(self, device=None):
        """Initialize with device (CPU or GPU)"""
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: Transformer models not available. Install transformers package.")
            self.device = None
        else:
            self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
        
        self.models = {}
        self.results = {}
        
    def train_bert(self, X_train, y_train, X_test, y_test, model_name='bert-base-uncased', epochs=3, batch_size=8):
        """Train and evaluate a BERT model"""
        if not TRANSFORMERS_AVAILABLE:
            print("Error: Transformers package not available. Cannot train BERT model.")
            return None
            
        print(f"\nTraining {model_name}...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        ).to(self.device)
        
        # Create datasets
        train_dataset = NewsDataset(X_train, y_train, tokenizer)
        test_dataset = NewsDataset(X_test, y_test, tokenizer)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f'./results/{model_name.split("/")[-1]}',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size*2,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )
        
        # Define compute metrics function
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted')
            return {'accuracy': accuracy, 'f1': f1}
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train model
        trainer.train()
        
        # Evaluate model
        eval_results = trainer.evaluate()
        
        # Get predictions
        test_predictions = trainer.predict(test_dataset)
        y_pred = test_predictions.predictions.argmax(-1)
        y_pred_labels = ['fake' if pred == 1 else 'real' for pred in y_pred]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred_labels, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred_labels)
        
        # Save model and results
        model_key = model_name.split('/')[-1]
        self.models[model_key] = {
            'model': model,
            'tokenizer': tokenizer,
            'trainer': trainer
        }
        
        self.results[model_key] = {
            'name': model_name.split('/')[-1].upper(),
            'accuracy': accuracy,
            'report': report,
            'y_pred': y_pred_labels,
            'y_prob': None,  # Would need additional code to extract probabilities
            'eval_results': eval_results,
            'color': colors['charts'][0]
        }
        
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        return self.results[model_key]
    
    def zero_shot_learning(self, texts, model_name="facebook/bart-large-mnli"):
        """Perform zero-shot learning for text classification"""
        if not TRANSFORMERS_AVAILABLE:
            print("Error: Transformers package not available. Cannot perform zero-shot learning.")
            return pd.DataFrame({'text': [], 'predicted': [], 'confidence': []})
            
        print(f"\nPerforming zero-shot learning with {model_name}...")
        
        # Load model for zero-shot classification
        classifier = pipeline("zero-shot-classification", model=model_name)
        
        # Define candidate labels
        candidate_labels = ["real news about health", "fake news about health"]
        
        # Process sample of texts (to avoid processing the entire dataset which could be slow)
        sample_size = min(100, len(texts))
        sample_texts = texts[:sample_size]
        
        results = []
        for text in sample_texts:
            result = classifier(text, candidate_labels)
            label = result['labels'][0]
            score = result['scores'][0]
            predicted_class = 'real' if 'real' in label else 'fake'
            results.append({
                'text': text[:100] + '...',  # Truncate for display
                'predicted': predicted_class,
                'confidence': score
            })
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Plot confidence distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(
            results_df, x='confidence', hue='predicted',
            palette={'real': colors['real_news'], 'fake': colors['fake_news']},
            bins=20, kde=True
        )
        plt.title('Zero-Shot Learning Confidence Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Confidence Score', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.legend(title='Predicted Class')
        plt.grid(alpha=0.3)
        plt.savefig('zero_shot_confidence.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results_df
    
    def few_shot_learning(self, X_train, y_train, X_test, label_examples=5, model_name="google/flan-t5-base"):
        """Demonstrate few-shot learning approach using a text generation model"""
        if not TRANSFORMERS_AVAILABLE:
            print("Error: Transformers package not available. Cannot perform few-shot learning.")
            return pd.DataFrame({'text': [], 'predicted': [], 'full_output': []})
            
        print(f"\nPerforming few-shot learning with {model_name}...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Prepare few-shot examples
        fake_examples = []
        real_examples = []
        
        for text, label in zip(X_train, y_train):
            if label == 'fake' and len(fake_examples) < label_examples:
                fake_examples.append(text)
            elif label == 'real' and len(real_examples) < label_examples:
                real_examples.append(text)
            
            if len(fake_examples) == label_examples and len(real_examples) == label_examples:
                break
        
        # Build prompt template with few-shot examples
        prompt_template = "Task: Classify the following text as 'real health news' or 'fake health news'.\n\nExamples of fake health news:\n"
        
        # Add fake examples
        for i, example in enumerate(fake_examples):
            prompt_template += f"{i+1}. \"{example[:100]}...\" - fake health news\n"
        
        prompt_template += "\nExamples of real health news:\n"
        
        # Add real examples
        for i, example in enumerate(real_examples):
            prompt_template += f"{i+1}. \"{example[:100]}...\" - real health news\n"
        
        prompt_template += "\nNow classify this text: \"{text}\"\nAnswer:"
        
        # Process sample of test texts
        sample_size = min(20, len(X_test))
        sample_texts = X_test[:sample_size]
        
        results = []
        for text in sample_texts:
            prompt = prompt_template.format(text=text[:200])
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Generate prediction
            outputs = model.generate(
                inputs["input_ids"], 
                max_length=10, 
                temperature=0.1,
                num_return_sequences=1
            )
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            
            # Extract classification from the model's output
            if "fake" in prediction:
                label = "fake"
            elif "real" in prediction:
                label = "real"
            else:
                label = "unclear"
            
            results.append({
                'text': text[:100] + '...',
                'predicted': label,
                'full_output': prediction
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Plot distribution of predictions
        plt.figure(figsize=(8, 6))
        prediction_counts = results_df['predicted'].value_counts()
        colors_map = {
            'real': colors['real_news'],
            'fake': colors['fake_news'],
            'unclear': '#CCCCCC'
        }
        
        sns.barplot(
            x=prediction_counts.index, 
            y=prediction_counts.values,
            palette=[colors_map.get(x, '#CCCCCC') for x in prediction_counts.index]
        )
        
        plt.title('Few-Shot Learning Predictions', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig('few_shot_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results_df

# Code to run this module if executed directly
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 100
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
    
    # Generate synthetic data
    texts = []
    for label in labels:
        if label == 'fake':
            title = np.random.choice(fake_titles)
        else:
            title = np.random.choice(real_titles)
            
        # Add some variation
        text = f"{title} {np.random.randint(1, 100)}"
        texts.append(text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Initialize Transformer models
    transformer = TransformerModels()
    
    # Demonstrate zero-shot learning (uncomment to run)
    # zero_shot_results = transformer.zero_shot_learning(X_test)
    
    # Demonstrate few-shot learning (uncomment to run)
    # few_shot_results = transformer.few_shot_learning(X_train, y_train, X_test)
    
    # For full BERT training, you'd need more data and GPU resources
    # bert_results = transformer.train_bert(X_train, y_train, X_test, y_test, epochs=1)  # Just 1 epoch for demo
    
    print("Transformer models demonstration complete")