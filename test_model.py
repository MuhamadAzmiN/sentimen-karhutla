import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('dataset_ulasan.csv')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stopwords = {'dan', 'yang', 'di', 'ini', 'itu', 'dengan', 'untuk', 'pada', 'ke', 'dari', 'sama', 'saya', 'saat', 'ada'}
    cleaned = [w for w in tokens if w not in stopwords]
    return ' '.join(cleaned)

df['clean_komentar'] = df['komentar'].apply(preprocess_text)
all_words = ' '.join(df['clean_komentar']).split()
counts = Counter(all_words)
print('Top word counts:')
for w in ['bagus', 'nyaman', 'puas', 'kurang', 'ramah', 'rusak', 'cantik']:
    print(f'{w}: {counts.get(w, 0)}')

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_komentar'], df['sentimen'], test_size=30, train_size=120, random_state=42, stratify=df['sentimen']
)
vec = CountVectorizer()
X_tr_vec = vec.fit_transform(X_train)
X_te_vec = vec.transform(X_test)

model = MultinomialNB()
model.fit(X_tr_vec, y_train)
preds = model.predict(X_te_vec)
acc = accuracy_score(y_test, preds)

print(f'Train shape: {X_tr_vec.shape}, Test shape: {X_te_vec.shape}')
print(f'Accuracy: {acc:.4f}')

test_sample = 'bagus'
test_pred = model.predict(vec.transform([preprocess_text(test_sample)]))[0]
print(f'Test prediction for "{test_sample}": {test_pred}')
