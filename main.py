import os
import requests
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def load_files_from_directory(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                processed_text = preprocess_text(text)
                texts.append(processed_text)
    return texts

def scrape_web_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    content = ' '.join([p.get_text() for p in paragraphs])
    return preprocess_text(content)

def train_model(texts, labels):
    vectorizer = TfidfVectorizer()
    classifier = LogisticRegression()
    model = make_pipeline(vectorizer, classifier)
    
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    return model, accuracy

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

# Example usage
texts = load_files_from_directory('path_to_directory')
web_content = scrape_web_content('https://example.com')
texts.append(web_content)

# Dummy labels for example purposes
labels = [0 if i < len(texts) // 2 else 1 for i in range(len(texts))]

model, accuracy = train_model(texts, labels)
print(f'Model accuracy: {accuracy}')

save_model(model, 'text_classification_model.pkl')
loaded_model = load_model('text_classification_model.pkl')
