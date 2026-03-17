import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
print("Loading dataset...")
data = pd.read_csv("dataset.csv")

print("\nDataset preview:")
print(data.head())

# Features and labels
X = data['text']
y = data['label']

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

print("\nPreprocessing text...")
X = X.apply(preprocess)

# Vectorization
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Prediction function
def predict_news(news):
    news = preprocess(news)
    news = vectorizer.transform([news])
    result = model.predict(news)

    if result[0].lower() == "fake":
        return "FAKE NEWS ❌"
    else:
        return "REAL NEWS ✅"

# ---------- INPUT LOOP (THIS IS THE PART YOU ARE MISSING) ----------

print("\nProgram ready for prediction...")

while True:
    text = input("\nEnter news text (or type exit): ")

    if text.lower() == "exit":
        print("Exiting program...")
        break

    prediction = predict_news(text)
    print("Prediction:", prediction)
