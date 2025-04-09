import pandas as pd
import numpy as np
from newspaper import Article

TARGET_WORDS = ["kills", "vaccine", "force", "death", "facebook"]
CSV_PATH = "project/dataset/NewsFakeCOVID-19.csv"
SAVE_PATH = "Target_Word_Counts.csv"
ROWS_TO_PROCESS = 10

df = pd.read_csv(CSV_PATH)
if ROWS_TO_PROCESS:
    df = df.head(ROWS_TO_PROCESS)

def get_article_text(url):
    if pd.isna(url):
        return ""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.lower()
    except:
        return ""

def count_words_in_text(text, word):
    return text.split().count(word.lower())

texts = df["news_url"].apply(get_article_text)
for word in TARGET_WORDS:
    df[f"count_{word}"] = texts.apply(lambda text: count_words_in_text(text, word))


df.to_csv(SAVE_PATH, index=False)
print(f"✅ Done! Results saved to: {SAVE_PATH}")
