from flask import Flask, render_template, request
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import logging

# Flask app setup
app = Flask(__name__)
app.logger.setLevel(logging.ERROR)

# Download NLTK resources if not already present
def download_nltk_resources():
    resources = {'punkt': 'tokenizers/punkt', 'stopwords': 'corpora/stopwords'}
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

download_nltk_resources()

# Slang normalization dictionary
slang_dict = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "g": "tidak", "kagak": "tidak", "enggak": "tidak",
    "bgt": "banget", "pd": "pada", "dr": "dari", "dlm": "dalam", "nya": "nya", "org": "orang", "jd": "jadi",
    "jg": "juga", "aja": "saja", "tp": "tapi", "sm": "sama", "udh": "sudah", "dl": "dulu", "mksh": "terima kasih",
    "gx": "tidak", "yg": "yang", "dgn": "dengan", "d": "di", "tdk": "tidak", "brg": "barang"
}

# Preprocessing function
def preprocess_text(text, slang_dict, stemmer):
    # Lowercasing
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and normalize slang words
    text = ' '.join([slang_dict.get(word, word) for word in word_tokenize(text)])
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Tokenization and stopwords removal
    stop_words = set(stopwords.words('indonesian')).union(set(stopwords.words('english')))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# Load trained SVM model and TF-IDF vectorizer
with open("svm_model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        preprocessed_message = preprocess_text(message, slang_dict, stemmer)
        vectorized_message = tfidf.transform([preprocessed_message])
        prediction = clf.predict(vectorized_message)

        # Mapping sentiment labels to categories
        label_mapping = {2: "Negatif", 0: "Netral", 1: "Positif"}
        result = label_mapping[prediction[0]]

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=False)
