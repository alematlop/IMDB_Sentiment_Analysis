import joblib
from preprocess import clean_text
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, find


def predict_sentiment(text):

    model = joblib.load("../models/logistic_regression.pkl")
    vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
    cleaned_text = clean_text(text)
    transformed_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_text)

    return "Positive" if prediction[0] == 2 else "Negative"

def get_LR_model_accuracy():

    df_test = pd.read_csv('../data/processed/test.csv')
    df_test['text'] = df_test['text'].apply(clean_text)

    vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
    X_test = vectorizer.transform(df_test['text'])
    Y_test = df_test['label']

    model = joblib.load("../models/logistic_regression.pkl")

    i = -1
    c = 0
    for el in X_test:

        i += 1

        if model.predict(el) == Y_test[i]:
            c += 1

    print(c/i)


# Get Model Accuracy Percentage
# get_LR_model_accuracy()

# Test Custom Input
sentiment = predict_sentiment("This movie was fantastic! I loved every moment.")
print(sentiment)


