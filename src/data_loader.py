import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.datasets import IMDB

train_data, test_data = IMDB(split=('train', 'test'))

train_texts, train_labels = zip(*[(text, label) for label, text in train_data])
test_texts, test_labels = zip(*[(text, label) for label, text in test_data])

df_train = pd.DataFrame({'text': train_texts, 'label': train_labels})
df_test = pd.DataFrame({'text': test_texts, 'label': test_labels})

df_train.to_csv("../data/processed/train.csv", index = False)
df_test.to_csv("../data/processed/test.csv", index = False)