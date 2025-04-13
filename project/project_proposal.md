# Detecting Health Misinformation: Beyond COVID-19

## Problem Overview

We plan to create a model that detects fake health news, primarily focusing on COVID-19 misinformation yet aiming to see if it generalizes to other misleading health claims. Fake content about cures or treatments can harm public well-being and erode trust in medical advice. By training on an extensive pool of labeled COVID-19 examples, we want to see whether our approach catches broader falsehoods, like bogus home remedies or miracle cures. If successful, this would demonstrate that our system is not simply memorizing pandemic-specific language, but actually learning meaningful patterns of health misinformation.

## Data

We will start with two main COVID-19 fake news datasets:

1. **COVID-19 Fake News Dataset** (Kaggle): Contains news articles with binary classification (fake/real).
2. **CoAID Collection**: Integrates articles, social media posts, and user engagement metrics with fake/real labels.

Both datasets provide a reliable foundation for training. After cleaning and merging these sources, we will create a smaller test set of non-COVID fake health claims, such as pseudoscientific "cures." 

**Key Challenges:**
- Managing noisy data from social media sources
- Addressing class imbalance (typically fewer "fake" examples)
- Preventing the model from overfitting to pandemic-specific terminology

## Method

Our implementation pipeline includes:

1. **Text Preprocessing**:
   - Tokenization and lemmatization
   - Lowercasing and stopword removal
   - Special handling for health-specific terminology

2. **Modeling Approaches**:
   - **Baseline**: Classical ML with TF-IDF features
     - Logistic Regression
     - Support Vector Machines (SVM)
   - **Advanced**: Transformer-based approaches
     - Fine-tuned BERT model
     - Domain adaptation techniques

Libraries including scikit-learn and PyTorch will power our implementation. The core research question is whether deeper models can capture subtle misinformation cues that transfer to novel, non-COVID health claims.

## Related Work

| Research | Key Contribution | Relation to Our Work |
|----------|------------------|----------------------|
| **Patwa et al. (2021)** | Introduced large-scale COVID-19 fake news dataset; found simpler models sometimes outperformed complex ones | Informs our baseline comparison strategy; we extend beyond COVID to test generalization |
| **Cui & Lee (2020)** | Created CoAID dataset combining articles, tweets, and engagement data | We leverage this dataset while focusing more on linguistic patterns than social metrics |
| **Shahi & Nandini (2020)** | Compiled multilingual FakeCovid with fact-checked articles | While we focus on English content, their cross-domain insights inform our transfer approach |
| **Kar et al. (2020)** | Implemented BERT-based detection for English and Indic languages | We adapt similar architecture focusing on domain transfer rather than language transfer |
| **Vijjali et al. (2020)** | Developed two-stage transformer for both detection and fact-checking | Our preprocessing incorporates health-specific terminology handling; we evaluate cross-domain performance |

## Evaluation

We will implement a comprehensive evaluation strategy:

1. **In-Domain Performance**:
   - Accuracy, precision, recall, and F1-score on COVID-19 data
   - Confusion matrices to identify error patterns
   - ROC and precision-recall curves

2. **Cross-Domain Evaluation**:
   - Apply trained models to non-COVID health claims
   - Measure performance degradation across domains
   - Analyze error cases to identify domain-specific challenges

3. **Baseline Comparisons**:
   - Simple majority-class predictor
   - Traditional ML approaches (logistic regression)
   - Feature importance analysis

If time permits, we will explore limited fine-tuning on out-of-domain examples to improve transferability and determine the minimum adaptation required for effective cross-domain performance.

## References

1. COVID-19 Fake News Dataset (Kaggle)  
   https://www.kaggle.com/datasets/arashnic/covid19-fake-news

2. CoAID – COVID-19 Healthcare Misinformation Dataset (GitHub)  
   https://github.com/cuilimeng/CoAID

3. Patwa et al. (2021) – "Fighting an Infodemic: COVID-19 Fake News Dataset"  
   https://arxiv.org/abs/2011.03327

4. Cui & Lee (2020) – "CoAID: COVID-19 Healthcare Misinformation Dataset"  
   https://arxiv.org/abs/2006.00885

5. Shahi & Nandini (2020) – "FakeCovid: A Multilingual Cross-Domain Fact Check News Dataset for COVID-19"  
   https://arxiv.org/abs/2006.11343

6. Kar et al. (2020) – "No Rumours Please! A Multi-Indic-Lingual Approach for COVID Fake-Tweet Detection"  
   https://arxiv.org/abs/2010.06906

7. Vijjali et al. (2020) – "Two Stage Transformer Model for COVID-19 Fake News Detection and Fact Checking"  
   https://arxiv.org/abs/2011.13253