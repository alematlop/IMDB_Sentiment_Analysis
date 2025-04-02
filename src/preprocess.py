import re
import string
import nltk
from nltk.corpus import stopwords

def clean_text(text):

    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text

