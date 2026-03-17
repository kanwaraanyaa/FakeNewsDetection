from flask import Flask, request, render_template_string
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Make sure stopwords are downloaded
nltk.download('stopwords')

app = Flask(__name__)

# ---------------- LOAD & TRAIN MODEL ONCE ----------------
print("Loading dataset...")
data = pd.read_csv("dataset.csv")

X = data['text']
y = data['label']

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

X = X.apply(preprocess)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model trained with accuracy:", accuracy)

# Prediction function
def predict_news(news):
    news = preprocess(news)
    news = vectorizer.transform([news])
    result = model.predict(news)

    if result[0].lower() == "fake":
        return "FAKE NEWS ❌"
    else:
        return "REAL NEWS ✅"

# ---------------- SIMPLE HTML TEMPLATE ----------------
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detection</title>
    <style>
        body { font-family: Arial; background: #0f172a; color: white; text-align: center; }
        .box { margin-top: 100px; background: #1e293b; padding: 30px; border-radius: 10px; width: 400px; margin-left: auto; margin-right: auto; }
        textarea { width: 100%; height: 120px; border-radius: 5px; padding: 10px; }
        button { margin-top: 15px; padding: 10px 20px; font-size: 16px; border: none; border-radius: 5px; background: #22c55e; color: black; cursor: pointer; }
        h2 { margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="box">
        <h2>Fake News Detection</h2>
        <form method="post">
            <textarea name="news" placeholder="Enter news text here...">{{news}}</textarea><br>
            <button type="submit">Check</button>
        </form>
        {% if result %}
            <h3>Prediction: {{result}}</h3>
        {% endif %}
        <p style="margin-top:20px; font-size:12px;">Model Accuracy: {{accuracy}}</p>
    </div>
</body>
</html>
"""

# ---------------- FLASK ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    news = ""
    if request.method == "POST":
        news = request.form["news"]
        result = predict_news(news)
    return render_template_string(HTML, result=result, news=news, accuracy=accuracy)

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)
