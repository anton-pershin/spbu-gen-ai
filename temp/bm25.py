import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

nltk.download("popular")
nltk.download("punkt")
nltk.download("stopwords")

stemmer = SnowballStemmer("russian")
text = "Листовые листочки лист листва листве почему так"

stop_words = set(stopwords.words('russian'))
filtered_tokens = [word for word in tokens if word not in stop_words]

print(filtered_tokens)

tokens = word_tokenize(text)
stemmed_words = [stemmer.stem(word) for word in tokens]
print(stemmed_words)
