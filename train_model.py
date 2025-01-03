import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import re
import string  # Pastikan modul 'string' diimpor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download stopwords and punkt if not already installed
nltk.download('stopwords')
nltk.download('punkt')

# Slang dictionary
new_slang_dict = {
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "g": "tidak",
    "kagak": "tidak",
    "enggak": "tidak",
    "bgt": "banget",
    "pd": "pada",
    "dr": "dari",
    "dlm": "dalam",
    "nya": "nya",
    "org": "orang",
    "jd": "jadi",
    "jg": "juga",
    "aja": "saja",
    "tp": "tapi",
    "sm": "sama",
    "udh": "sudah",
    "dl": "dulu",
    "mksh": "terima kasih",
    "gk": "tidak",
    "yg": "yang",
    "dgn": "dengan",
    "d": "di",
    "tdk": "tidak",
    "brg": "barang"
}

# Slang normalization
def normalize_slang(text, slang_dict):
    words = word_tokenize(text)
    return ' '.join([slang_dict.get(word, word) for word in words])

# Text preprocessing
def preprocess_text(text, slang_dict, stemmer):
    text = text.lower()  # Lowercasing
    text = normalize_slang(text, slang_dict)  # Normalisasi slang
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca
    tokens = word_tokenize(text)
    text = re.sub(r'[^\w\s]', '', text)  # Removes punctuation, emoticons, and special characters
    text = re.sub(r'@\w+\s*', '', text)  # Remove mentions
    text = re.sub(r'https?://\S+', '', text)  # Remove links
    text = re.sub(r'\[.*?\]', ' ', text)  # Gunakan raw string 'r'
    text = re.sub(r'\(.*?\)', ' ', text)  # Gunakan raw string 'r'
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)  # Gunakan raw string 'r'
    text = re.sub(r'[‘’“”…♪♪]', '', text)  # Remove additional punctuations or non-sensical text
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'b ', ' ', text)
    text = re.sub(r'rt ', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces

    # Menghapus stopwords
    custom_stopwords = set(nltk.corpus.stopwords.words('indonesian')).union(set(nltk.corpus.stopwords.words('english')))
    tokens = [word for word in tokens if word not in custom_stopwords]

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load Indonesian sentiment lexicon
lexicon_url = 'https://drive.google.com/file/d/1qPX0Uej3PqUQUI3op_oeEr8AdmrgOT2V/view?usp=sharing'
lexicon_path = 'https://drive.google.com/uc?export=download&id=' + lexicon_url.split('/')[-2]
df_senti = pd.read_csv(lexicon_path, sep=':', names=['word', 'value'])

# Konversi lexicon sentimen ke dalam dictionary
senti_dict = {df_senti.iloc[i]['word']: float(df_senti.iloc[i]['value']) for i in range(len(df_senti))}

# Inisialisasi SentimentIntensityAnalyzer dan tambahkan lexicon bahasa Indonesia
senti_indo = SentimentIntensityAnalyzer()
senti_indo.lexicon.update(senti_dict)

# Fungsi untuk menghitung skor sentimen
def compute_sentiment_scores(text):
    tokens = word_tokenize(text)
    scores = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "compound": 0
    }
    for word in tokens:
        score = senti_indo.polarity_scores(word)
        scores["positive"] += score['pos']
        scores["negative"] += score['neg']
        scores["neutral"] += score['neu']
        scores["compound"] += score['compound']

    # Skor sentimen keseluruhan untuk teks
    text_score = senti_indo.polarity_scores(text)
    scores["text_positive"] = text_score['pos']
    scores["text_negative"] = text_score['neg']
    scores["text_neutral"] = text_score['neu']
    scores["text_compound"] = text_score['compound']
    return scores

# Dataset preparation
df = pd.read_csv("data_real2.csv", encoding="latin-1")
df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'Kategori',
       'Nama Produk', 'Id Produk', 'Terjual', 'Id_Toko', 'Url', 'label'], inplace=True, errors='ignore')

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Apply preprocessing
df['text_clean_preprocessed'] = df['Ulasan'].apply(lambda x: preprocess_text(str(x), new_slang_dict, stemmer))

# Sentiment labeling using lexicon-based scoring
label_lexicon = []
for index, row in df.iterrows():
    score = compute_sentiment_scores(row['text_clean_preprocessed'])
    if score['text_compound'] >= 0.05:
        label_lexicon.append(1)  # Positif
    elif score['text_compound'] <= -0.05:
        label_lexicon.append(2)  # Negatif
    else:
        label_lexicon.append(0)  # Netral

# Add sentiment labels to dataframe
df['label_sentiment'] = label_lexicon

# Save labeled data
df.to_csv('data_lexicon_labeled.csv', index=False)

# Tfidf Vectorization
tfidf = TfidfVectorizer(max_df=0.5, min_df=2)
X = tfidf.fit_transform(df['text_clean_preprocessed'])
y = df['label_sentiment']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM model
clf = SVC(kernel='linear')
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
