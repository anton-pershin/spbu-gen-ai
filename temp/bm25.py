import regex

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    stemmer = SnowballStemmer("russian")
    stop_words = set(stopwords.words('russian'))
    text_wo_punct = regex.sub(r"[^\p{L}|\p{N}]+", " ", text)
    tokens = word_tokenize(text_wo_punct)
    tokens = [word for word in tokens if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return stemmed_words


if __name__ == "__main__":
    nltk.download("popular")
    nltk.download("punkt_tab")
    nltk.download("stopwords")

    df = pd.read_csv("~/temp/embed/toy_dataset.csv")
    docs = df.text.tolist()
    tokenized_docs = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)

    for d in tokenized_docs:
        scores = bm25.get_scores(d)
        pass
        
