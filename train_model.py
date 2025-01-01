import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

# Slang normalization dictionary
slang_dict = {
    "adlh": "adalah",
    "bgmn": "bagaimana",
    "nih": "ini",
    "afaik": "as far as I know",
    "abis": "habis",
    "ga": "tidak",
    "dikerjai": "dikerjakan"
}

# Preprocessing function
def preprocess_text(text, slang_dict, stemmer):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = ' '.join([slang_dict.get(word, word) for word in word_tokenize(text)])  # Slang normalization
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))  # Stopword list in Indonesian
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return ' '.join(tokens)

# Dataset preparation
df = pd.read_csv("data_real2.csv", encoding="latin-1")
df = df[['Ulasan', 'Rating']]

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Apply preprocessing
df['cleaned_text'] = df['Ulasan'].fillna("").apply(lambda x: preprocess_text(str(x), slang_dict, stemmer))

# Label mapping: 1.0-2.0 = negative (0), 3.0 = neutral (1), 4.0-5.0 = positive (2)
df['label'] = df['Rating'].map({
    1.0: 0, 2.0: 0,  # Negative
    3.0: 1,  # Neutral
    4.0: 2, 5.0: 2   # Positive
})

# Tfidf Vectorization
tfidf = TfidfVectorizer(max_df=0.5, min_df=2)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['label']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM model for multi-class classification
clf = SVC(kernel='linear', decision_function_shape='ovr')  # One-vs-Rest strategy
print("Training model...")
clf.fit(X_train, y_train)

# Save model and vectorizer
with open("svm_model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)
    print("Model saved as 'svm_model.pkl'")

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)
    print("Vectorizer saved as 'tfidf_vectorizer.pkl'")

# Training and saving finished
print("Training completed successfully!")
