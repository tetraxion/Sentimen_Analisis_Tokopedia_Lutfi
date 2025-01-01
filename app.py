from flask import Flask, render_template, request
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import nltk
import logging

# Flask setup
app = Flask(__name__)
app.logger.setLevel(logging.ERROR)  # Set logging level to ERROR for production

# Download NLTK resources if not already present
def download_nltk_resources():
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords'
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

download_nltk_resources()

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

# Load trained model and vectorizer
clf = pickle.load(open("svm_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

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
        
        # Mapping sentiment labels to the categories
        label_mapping = {0: "Negatif", 1: "Netral", 2: "Positif"}
        result = label_mapping[prediction[0]]  # Return the label based on prediction
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=False)  # Set debug=False for production
