import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Load your datasets
fake = pd.read_csv('fake.csv')
real = pd.read_csv('real.csv')
fake['label'] = 0
real['label'] = 1
df = pd.concat([fake, real]).reset_index(drop=True)

# NLP Pipeline
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
x = tfidf.fit_transform(df['text'])
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train Models
lr = LogisticRegression().fit(x_train, y_train)
nb = MultinomialNB().fit(x_train, y_train)

# SAVE THE FILES (This fixes the error)
with open('vectorizer.pkl', 'wb') as f: pickle.dump(tfidf, f)
with open('model_lr.pkl', 'wb') as f: pickle.dump(lr, f)
with open('model_nb.pkl', 'wb') as f: pickle.dump(nb, f)

print("Success: .pkl files created!")
