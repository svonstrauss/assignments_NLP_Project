Problem Overview
We plan to create a model that detects fake health news, primarily focusing on COVID-19 misinformation yet aiming to see if it generalizes to other misleading health claims. Fake content about cures or treatments can harm public well-being and erode trust in medical advice. By training on an extensive pool of labeled COVID-19 examples, we want to see whether our approach catches broader falsehoods, like bogus home remedies or miracle cures. If successful, this would demonstrate that our system is not simply memorizing pandemic-specific language, but actually learning meaningful patterns of health misinformation.

Data
We will start with two main COVID-19 fake news datasets: one from Kaggle (the “COVID-19 Fake News Dataset”) and the CoAID collection (combining articles, social media, and user engagement data). Both sets label items as “fake” or “real,” providing a reliable foundation for training. After cleaning and merging these, we will create a smaller test set of non-COVID fake health claims, such as pseudoscientific “cures.” The main challenges include noisy data from social media, class imbalance (often fewer “fake” examples), and ensuring the model doesn’t get stuck on pandemic-only terms.

Method
Our pipeline will begin with standard text preprocessing, including tokenization, lowercasing, and possibly removing stopwords. We will compare a baseline (e.g., a logistic regression or SVM with TF-IDF features) to a more advanced transformer-based classifier like a fine-tuned BERT model. Libraries like scikit-learn and PyTorch will be used for implementation. The key question is whether a deeper model can capture more subtle cues of misinformation and then transfer those insights to novel, non-COVID claims.

Related Work

Patwa et al. (2021): They introduced a large-scale COVID-19 fake news dataset and tested classical machine learning models like SVM and logistic regression. Their surprising finding was that simpler models sometimes outperformed more complex ones. This emphasizes the importance of baseline comparisons when studying misinformation.
 

Cui & Lee (2020): Their CoAID dataset brings together articles, tweets, and user engagement data on COVID-19 misinformation. A unique aspect is capturing how social interactions (likes, shares) correlate with fake news spread. That makes CoAID useful for studying the viral potential of health misinformation.
 

Shahi & Nandini (2020): They compiled FakeCovid, a multilingual resource with fact-checked COVID-19 articles from dozens of countries. Unlike others, their focus is on cross-lingual and cross-domain analysis, highlighting how fake news can vary in different cultural settings. Their results showed challenges in detecting false content in multiple languages, but also offered insights into bridging language gaps.
 

Kar et al. (2020): This study tackled multilingual fake tweet detection (in English and Indic languages) using a BERT-based framework. Their work demonstrated that even limited labeled data can be effective with the right pretrained model. It also highlighted that domain-specific features (like user metadata) can further boost detection accuracy.
 

Vijjali et al. (2020): They proposed a two-stage transformer method for detecting and fact-checking COVID-19 fake news. First, relevant facts are retrieved from a knowledge base, and second, the model performs textual entailment to verify claims. It stands out by combining misinformation detection with an automated fact-check step.
Evaluation
We will measure accuracy, precision, recall, and F1-score on a held-out portion of the COVID-19 data to gauge in-domain performance. Our baseline will be a simple classifier (e.g., a majority-class predictor or logistic regression) against which we compare the more advanced models. Then, we will apply the trained model to our smaller set of non-COVID health claims. Any significant performance drop here indicates overfitting to COVID-specific language. We will present confusion matrices to clarify where errors occur, and possibly plot ROC or precision-recall curves. If time permits, we may explore limited fine-tuning on out-of-domain examples to see if that improves transferability.

References
COVID-19 Fake News Dataset (Kaggle)
https://www.kaggle.com/datasets/arashnic/covid19-fake-newsLinks to an external site.

CoAID – COVID-19 Healthcare Misinformation Dataset (GitHub)
https://github.com/cuilimeng/CoAIDLinks to an external site.

Patwa et al. (2021) – “Fighting an Infodemic: COVID-19 Fake News Dataset”
https://arxiv.org/abs/2011.03327Links to an external site.

Cui & Lee (2020) – “CoAID: COVID-19 Healthcare Misinformation Dataset”
https://arxiv.org/abs/2006.00885Links to an external site.

Shahi & Nandini (2020) – “FakeCovid: A Multilingual Cross-Domain Fact Check News Dataset for COVID-19”
https://arxiv.org/abs/2006.11343Links to an external site.

Kar et al. (2020) – “No Rumours Please! A Multi-Indic-Lingual Approach for COVID Fake-Tweet Detection”
https://arxiv.org/abs/2010.06906Links to an external site.

Vijjali et al. (2020) – “Two Stage Transformer Model for COVID-19 Fake News Detection and Fact Checking”
https://arxiv.org/abs/2011.13253Links to an external site.

