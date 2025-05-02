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
import random
from nltk.corpus import wordnet

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
    
    # Import explainability libraries
    try:
        import shap
        from captum.attr import LayerIntegratedGradients, visualization
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        EXPLAINABILITY_AVAILABLE = True
    except ImportError:
        print("Warning: Explainability libraries (shap, captum) not available. Explanations will be limited.")
        EXPLAINABILITY_AVAILABLE = False
except ImportError as e:
    print(f"Warning: Could not import transformer libraries - {str(e)}")
    TRANSFORMERS_AVAILABLE = False
    EXPLAINABILITY_AVAILABLE = False
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
    """Class for training and evaluating transformer models with explainability"""
    
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
        
        # Set up color maps for visualizations
        if EXPLAINABILITY_AVAILABLE:
            # Define a custom colormap for attention visualization
            self.attention_cmap = LinearSegmentedColormap.from_list(
                'attention_cmap', 
                ['#F1FAEE', '#E63946']  # Light to red for attention
            )

    def synonym_swap(text, swap_prob=0.15):
        words = text.split()
        new_words = []
        for word in words:
            if random.random() < swap_prob:
                synonyms = wordnet.synsets(word)
                if synonyms:
                    synonym_words = [lemma.name().replace("_", " ") for lemma in synonyms[0].lemmas() if lemma.name().lower() != word.lower()]
                    if synonym_words:
                        new_word = random.choice(synonym_words)
                        new_words.append(new_word)
                        continue
            new_words.append(word)
        return " ".join(new_words)
        
    def train_bert(self, X_train, y_train, X_test, y_test, model_name='bert-base-uncased', epochs=3, batch_size=8, check_leakage=True, fuzzy_threshold=0.9, augment=True):
        """Train and evaluate a BERT model with data leakage checking and optional synonym swapping"""
        if not TRANSFORMERS_AVAILABLE:
            print("Error: Transformers package not available. Cannot train BERT model.")
            return None

        print(f"\nTraining {model_name}...")

        # Optional Data Leakage Check
        if check_leakage:
            from difflib import SequenceMatcher
            leak_count = 0
            for val_text in X_test:
                for train_text in X_train:
                    sim = SequenceMatcher(None, val_text.strip().lower(), train_text.strip().lower()).ratio()
                    if sim >= fuzzy_threshold:
                        leak_count += 1
                        break
            if leak_count > 0:
                print(f"WARNING: Detected {leak_count} potential duplicate or near-duplicate samples between training and validation!")

        # Apply synonym swap augmentation
        if augment:
            X_train = [synonym_swap(text) for text in X_train]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)

        train_dataset = NewsDataset(X_train, y_train, tokenizer)
        test_dataset = NewsDataset(X_test, y_test, tokenizer)

        training_args = TrainingArguments(
            output_dir=f'./results/{model_name.split("/")[-1]}',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted')
            return {'accuracy': accuracy, 'f1': f1}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        test_predictions = trainer.predict(test_dataset)
        y_pred = test_predictions.predictions.argmax(-1)
        y_pred_labels = ['fake' if pred == 1 else 'real' for pred in y_pred]

        report = classification_report(y_test, y_pred_labels, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred_labels)

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
            'y_prob': None,
            'eval_results': eval_results,
            'color': colors['charts'][0]
        }

        print(f"{model_name} Accuracy: {accuracy:.4f}")
        
        # Check for suspiciously high performance
        if accuracy > 0.95:
            print("\n⚠️ WARNING: Suspiciously high accuracy detected!")
            print(f"Accuracy of {accuracy:.4f} may indicate data leakage or overfitting.")
            
            # Perform additional checks for potential problems
            perfect_predictions = (y_test == y_pred_labels).mean()
            if perfect_predictions > 0.95:
                print(f"- {perfect_predictions*100:.1f}% of predictions are correct")
                
            # Count extremely confident predictions
            if hasattr(test_predictions, 'predictions'):
                confidence = np.max(torch.nn.functional.softmax(torch.tensor(test_predictions.predictions), dim=1).numpy(), axis=1)
                high_confidence = (confidence > 0.99).mean()
                if high_confidence > 0.8:  # If more than 80% of predictions have >99% confidence
                    print(f"- {high_confidence*100:.1f}% of predictions have >99% confidence")
                    print("This could indicate that the model is memorizing the training data.")
            
            print("\nRecommendations:")
            print("1. Verify your validation data doesn't overlap with training data")
            print("2. Use a more challenging test set")
            print("3. Evaluate on data from different domains/sources")
            print("4. Try reducing model capacity or adding regularization")
            
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
        
    def visualize_attention(self, model_key, text, target_class='fake', layer_idx=-1, head_idx=None):
        """
        Visualize attention patterns in transformer models
        
        Args:
            model_key: Name of model to explain
            text: Text input to explain
            target_class: 'fake' or 'real'
            layer_idx: Which transformer layer to visualize
            head_idx: Which attention head to visualize (None for all heads)
        
        Returns:
            None, displays visualization
        """
        if not TRANSFORMERS_AVAILABLE or not EXPLAINABILITY_AVAILABLE:
            print("Error: Transformer or explainability libraries not available.")
            return
        
        if model_key not in self.models:
            print(f"Model {model_key} not found. Train or load a model first.")
            return
        
        model = self.models[model_key]['model']
        tokenizer = self.models[model_key]['tokenizer']
        
        # Preprocess input
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_words = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Get model outputs and attention weights
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions
            logits = outputs.logits
            predicted = torch.argmax(logits).item()
            probs = torch.nn.functional.softmax(logits, dim=1)
            target_idx = 1 if target_class == 'fake' else 0
            
            print(f"Model prediction: {'Fake' if predicted == 1 else 'Real'}")
            print(f"Confidence: {probs[0][predicted]:.4f}")
        
        # Get attention weights from specified layer
        attn_layer = attentions[layer_idx]  # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        # If head_idx specified, get specific head, otherwise average across heads
        if head_idx is not None:
            attn_weights = attn_layer[0, head_idx].cpu().numpy()  # Shape: (seq_len, seq_len)
            title = f"Attention Weights - Layer {layer_idx}, Head {head_idx}"
        else:
            attn_weights = attn_layer[0].mean(dim=0).cpu().numpy()  # Shape: (seq_len, seq_len)
            title = f"Average Attention Weights - Layer {layer_idx}"
        
        # Plot attention weights
        plt.figure(figsize=(10, 10))
        plt.imshow(attn_weights, cmap=self.attention_cmap)
        plt.colorbar(label="Attention Weight")
        
        # Add token labels
        plt.xticks(range(len(token_words)), token_words, rotation=90)
        plt.yticks(range(len(token_words)), token_words)
        
        # Highlight target tokens
        valid_tokens = attention_mask[0].sum().item()
        plt.xlim(-0.5, valid_tokens - 0.5)
        plt.ylim(valid_tokens - 0.5, -0.5)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'attention_visualization_{model_key}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate a more readable attention-highlighted text view
        self._visualize_token_attributions(model_key, text, attn_weights, token_words)
        
        return
    
    def _visualize_token_attributions(self, model_key, text, attention_weights, tokens):
        """Create a visualization of token attributions from attention"""
        if model_key not in self.models:
            print(f"Model {model_key} not found. Cannot visualize token attributions.")
            return
            
        tokenizer = self.models[model_key]['tokenizer']
        
        plt.figure(figsize=(12, 5))
        
        # Get token importance (sum of attention received)
        token_importance = attention_weights.sum(axis=0)
        token_importance = token_importance / token_importance.max()  # Normalize
        
        # Break into lines for readability
        words = text.split()
        max_words_per_line = 15
        lines = [words[i:i+max_words_per_line] for i in range(0, len(words), max_words_per_line)]
        
        # Plot each line separately
        for line_idx, line in enumerate(lines):
            y_pos = line_idx * 0.1
            x_pos = 0
            
            for word in line:
                # Find token indices for this word (handles subword tokenization)
                word_tokens = tokenizer.tokenize(word)
                if not word_tokens:
                    continue  # Skip if no tokens found
                
                # Find token importance - use max for multi-token words
                word_importances = []
                word_token_idx = None
                
                for token in word_tokens:
                    if token in tokens:
                        token_idx = tokens.index(token)
                        word_importances.append(token_importance[token_idx])
                
                if word_importances:
                    word_importance = max(word_importances)
                else:
                    word_importance = 0.1  # Default low importance
                
                # Add the word with color based on importance
                plt.text(x_pos, y_pos, word + " ", 
                         color=plt.cm.RdYlBu_r(word_importance),
                         fontsize=14, fontweight='bold' if word_importance > 0.5 else 'normal')
                
                x_pos += len(word) * 0.02 + 0.02  # Space between words
            
        plt.xlim(0, 1)
        plt.ylim(-0.1, len(lines) * 0.1 + 0.1)
        plt.axis('off')
        plt.title(f"Attention Highlights - {model_key}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'attention_highlights_{model_key}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def explain_with_shap(self, model_key, texts, target_class='fake', max_samples=20):
        """
        Generate SHAP explanations for transformer model predictions
        
        Args:
            model_key: Name of model to explain
            texts: List of texts to explain
            target_class: 'fake' or 'real'
            max_samples: Maximum number of samples to explain
        
        Returns:
            None, displays SHAP visualizations
        """
        if not TRANSFORMERS_AVAILABLE or not EXPLAINABILITY_AVAILABLE:
            print("Error: Transformer or explainability libraries not available.")
            return
        
        try:
            # For SHAP, we need to ensure we have shap library
            import shap
        except ImportError:
            print("Error: SHAP library not available. Please install with 'pip install shap'.")
            return
        
        if model_key not in self.models:
            print(f"Model {model_key} not found. Train or load a model first.")
            return
        
        model = self.models[model_key]['model']
        tokenizer = self.models[model_key]['tokenizer']
        
        # Limit the number of samples to explain
        num_samples = min(max_samples, len(texts))
        sample_texts = texts[:num_samples]
        
        # Create a text explainer
        print(f"Creating SHAP explainer for {model_key}...")
        
        # Define a prediction function for SHAP
        def predict(texts):
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                return probs.cpu().numpy()
        
        # Use the Partition explainer for text
        explainer = shap.Explainer(predict, tokenizer)
        
        # Generate explanations
        print("Generating SHAP explanations (this may take a while)...")
        shap_values = explainer(sample_texts)
        
        # Get class index for target class
        class_idx = 1 if target_class == 'fake' else 0
        
        # Plot SHAP summary
        plt.figure(figsize=(12, 8))
        shap.plots.text(shap_values[:, :, class_idx], display=False)
        plt.title(f"SHAP Explanation for '{target_class.capitalize()}' Class - {model_key}")
        plt.tight_layout()
        plt.savefig(f'shap_explanation_{model_key}_{target_class}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # For the first few samples, show detailed explanations
        for i in range(min(3, num_samples)):
            plt.figure(figsize=(12, 4))
            shap.plots.text(shap_values[i, :, class_idx], display=False)
            plt.title(f"SHAP Explanation for Sample {i+1}")
            plt.tight_layout()
            plt.savefig(f'shap_sample_{i+1}_{model_key}.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return shap_values
    
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
    
    print("\n===== TRANSFORMER MODELS WITH EXPLAINABILITY =====")
    
    # Demonstration choice
    DEMO_MODE = "explainability"  # Options: "training", "zero_shot", "few_shot", "explainability"
    
    if DEMO_MODE == "training":
        # For full BERT training (requires GPU resources)
        print("\nTraining BERT model with data leakage detection and synonym swapping...")
        bert_results = transformer.train_bert(
            X_train, y_train, X_test, y_test, 
            epochs=1,  # Just 1 epoch for demo
            check_leakage=True,
            augment=True
        )
        
    elif DEMO_MODE == "zero_shot":
        # Demonstrate zero-shot learning
        print("\nPerforming zero-shot learning on test data...")
        zero_shot_results = transformer.zero_shot_learning(X_test)
        
    elif DEMO_MODE == "few_shot":
        # Demonstrate few-shot learning
        print("\nPerforming few-shot learning with example COVID headlines...")
        few_shot_results = transformer.few_shot_learning(X_train, y_train, X_test)
        
    elif DEMO_MODE == "explainability":
        # For explainability demos, use a pre-trained model
        # In a real implementation, you'd load a fine-tuned model
        
        print("\nDemonstrating model explainability techniques...")
        print("Note: For a real implementation, you would use your fine-tuned model")
        
        if TRANSFORMERS_AVAILABLE and EXPLAINABILITY_AVAILABLE:
            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            
            # Store model and tokenizer for explainability demos
            transformer.models["distilbert"] = {
                "model": model,
                "tokenizer": tokenizer,
                "trainer": None
            }
            
            # Example fake news headline to explain
            fake_example = "SHOCKING: 5G towers have been proven to spread COVID and the government is hiding it from citizens!"
            
            # 1. Attention visualization
            print("\nVisualizing attention patterns for a fake news headline...")
            
            try:
                # This is just a demo - in real usage, you'd check if output_attentions=True is supported
                transformer.visualize_attention("distilbert", fake_example)
                
                # 2. SHAP explanations (if available)
                print("\nGenerating SHAP explanations for sample headlines...")
                sample_texts = [
                    "SHOCKING: 5G towers have been proven to spread COVID and the government is hiding it from citizens!",
                    "New study shows masks are effective at reducing COVID-19 transmission rates in indoor settings.",
                    "Vaccines have dangerous ingredients that the FDA has hidden from the public records!"
                ]
                
                # Note: SHAP may not work with all pretrained models without fine-tuning
                # transformer.explain_with_shap("distilbert", sample_texts)
                
                print("\nNote: For explainability to work well in your application:")
                print("1. Fine-tune your model on your specific dataset")
                print("2. Install shap and captum libraries")
                print("3. Run attention visualization on specific examples")
                print("4. Use SHAP for deeper understanding of model decisions")
            
            except Exception as e:
                print(f"Error in explainability demo: {str(e)}")
                print("This is expected in the demo - for real usage, fine-tune a model first")
        else:
            print("Explainability libraries not available. Install transformers, shap, and captum.")
    
    print("\nTransformer models demonstration complete")