import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# 1. Muat data
# Pastikan fail fake.csv dan real.csv ada dalam folder yang sama
df_fake = pd.read_csv('fake.csv')
df_real = pd.read_csv('real.csv')
df_fake['label'] = 0
df_real['label'] = 1
df = pd.concat([df_fake, df_real]).reset_index(drop=True)

# 2. Pra-pemprosesan
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# 3. Vectorization (Satu vectorizer untuk kedua-dua model)
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train_vec = tfidf.fit_transform(x_train)

# 4. Latih Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(x_train_vec, y_train)

# 5. Latih Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(x_train_vec, y_train)

# 6. Simpan fail (Penting untuk elakkan FileNotFoundError)
with open('vectorizer.pkl', 'wb') as f: pickle.dump(tfidf, f)
with open('model_lr.pkl', 'wb') as f: pickle.dump(lr_model, f)
with open('model_nb.pkl', 'wb') as f: pickle.dump(nb_model, f)

print("Semua model berjaya disimpan!")
