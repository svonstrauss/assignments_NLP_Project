import re
import random
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure required nltk resources
for resource in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text, remove_stopwords=True):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "[URL]", text)
    text = re.sub(r"\S+@\S+", "[EMAIL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    # Health replacements
    health_replacements = {
        "vaxx": "vaccine",
        "vaxxed": "vaccinated",
        "antivaxx": "antivaccine",
        "jab": "vaccine",
        "vax": "vaccine",
    }

    for original, replacement in health_replacements.items():
        text = re.sub(r"\b" + original + r"\b", replacement, text)

    tokens = word_tokenize(text)
    if remove_stopwords:
        clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    else:
        clean_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(clean_tokens)

def synonym_swap(text, swap_prob=0.1):
    words = text.split()
    new_words = []
    for word in words:
        if random.random() < swap_prob and word.isalpha() and word not in stop_words:
            synonyms = wordnet.synsets(word)
            synonym_candidates = []
            for syn in synonyms:
                for lemma in syn.lemmas():
                    candidate = lemma.name().replace("_", " ")
                    if candidate.lower() != word.lower():
                        synonym_candidates.append(candidate)
            if synonym_candidates:
                new_word = random.choice(synonym_candidates)
                new_words.append(new_word)
                continue
        new_words.append(word)
    return " ".join(new_words)

def detect_near_duplicates(train_texts, val_texts, threshold=0.9):
    leaks = []
    for val in val_texts:
        for train in train_texts:
            sim = SequenceMatcher(None, val, train).ratio()
            if sim >= threshold:
                leaks.append((train, val))
                break
    return leaks
