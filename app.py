from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)

# Add cache busting
import time
app.config['VERSION'] = str(int(time.time()))

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


def clean_text(text):
    text = str(text).lower()
    # Remove brackets and content
    text = re.sub(r'\[.*?\]', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)
    # Remove extra whitespace but keep structure
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove single characters
    text = re.sub(r'\b\w\b', '', text)
    return text.strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form.get("news", "").strip()

    if not news:
        return render_template("index.html", error="Please enter a news article to analyze.")

    # Text statistics
    word_count = len(news.split())
    char_count = len(news)
    sentence_count = max(1, len(re.split(r'[.!?]+', news.strip())) - 1)

    # Check minimum length
    if word_count < 10:
        return render_template("index.html", error="Please enter at least 10 words for analysis.", news_text=news)
    
    # Warning for short articles
    short_article_warning = None
    if word_count < 100:
        short_article_warning = f"⚠️ This text is very short ({word_count} words). For best accuracy, please paste the full article (at least 100+ words). Short headlines or snippets may not be classified accurately."

    # Clean, vectorize, predict
    cleaned = clean_text(news)
    data = vectorizer.transform([cleaned])
    prediction = model.predict(data)[0]

    # Get probability estimates from the ensemble
    try:
        # Get probabilities from soft voting ensemble
        probabilities = model.predict_proba(data)[0]
        prob_fake = probabilities[0]
        prob_real = probabilities[1]
        
        # Use a more conservative threshold instead of simple majority
        # Only classify as FAKE if we're very confident (>70%)
        # This reduces false positives on unfamiliar content
        confidence_threshold = 0.70
        
        if prob_real > confidence_threshold:
            result = "REAL"
            label = "Real News"
            confidence = round(prob_real * 100, 1)
        elif prob_fake > confidence_threshold:
            result = "FAKE"
            label = "Fake News"
            confidence = round(prob_fake * 100, 1)
        else:
            # Uncertain - show whichever is higher but flag as uncertain
            if prob_real > prob_fake:
                result = "UNCERTAIN_REAL"
                label = "Likely Real (Uncertain)"
                confidence = round(prob_real * 100, 1)
            else:
                result = "UNCERTAIN_FAKE"
                label = "Possibly Fake (Uncertain)"
                confidence = round(prob_fake * 100, 1)
        
        # Reduce confidence for very short articles (model is less reliable)
        if word_count < 100:
            confidence = max(51.0, confidence * 0.85)  # Reduce confidence by 15%
            confidence = round(confidence, 1)
            
    except AttributeError:
        # Fallback for models without predict_proba
        result = "REAL" if prediction == 1 else "FAKE"
        label = "Real News" if prediction == 1 else "Fake News"
        confidence = 75.0

    return render_template(
        "index.html",
        result=result,
        label=label,
        confidence=confidence,
        word_count=word_count,
        char_count=char_count,
        sentence_count=sentence_count,
        news_text=news,
        short_article_warning=short_article_warning
    )


if __name__ == "__main__":
    app.run(debug=False)
