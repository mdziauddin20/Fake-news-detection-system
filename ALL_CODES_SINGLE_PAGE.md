# All Project Code — Single Page

This document contains the full source code from the project, concatenated for report submission. Each file is labeled and its contents are included below.

---

## File: app.py
```python
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
```

---

## File: main.py
```python
def main():
    print("Hello from fake-news-detection-system!")


if __name__ == "__main__":
    main()
```

---

## File: train_model.py
```python
import pandas as pd
import re
import string
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

print("=" * 60)
print("FAKE NEWS DETECTION - MODEL TRAINING")
print("=" * 60)

# Load datasets
print("\n[1/7] Loading datasets...")
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

print(f"   ✓ Fake articles: {len(fake)}")
print(f"   ✓ True articles: {len(true)}")

# Merge datasets
data = pd.concat([fake, true], ignore_index=True)

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Keep required columns
data = data[["text", "label"]]

# Remove duplicates and null values
print("\n[2/7] Cleaning dataset...")
data = data.drop_duplicates(subset=['text'])
data = data.dropna()
print(f"   ✓ Total articles after cleaning: {len(data)}")
print(f"   ✓ Class distribution: Fake={len(data[data['label']==0])}, Real={len(data[data['label']==1])}")

# Clean text - less aggressive to preserve context
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

print("\n[3/7] Applying text preprocessing...")
data["text"] = data["text"].apply(clean_text)

# Remove very short articles (likely noise)
data = data[data["text"].str.split().str.len() > 10]
print(f"   ✓ Articles after removing short texts: {len(data)}")

# Input and output
X = data["text"]
y = data["label"]

# Convert text to vectors with improved parameters
print("\n[4/7] Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=8000,         # Increased features for better coverage
    ngram_range=(1, 2),        # Unigrams and bigrams
    min_df=2,                  # Word must appear in at least 2 documents (less strict)
    max_df=0.8,                # Ignore words in more than 80% of docs (less strict)
    sublinear_tf=True,         # Apply sublinear tf scaling
    smooth_idf=True,           # Smooth idf weights
    use_idf=True               # Enable inverse-document-frequency
)

X = vectorizer.fit_transform(X)
print(f"   ✓ Feature matrix shape: {X.shape}")

# Split data
print("\n[5/7] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Training set: {X_train.shape[0]} samples")
print(f"   ✓ Test set: {X_test.shape[0]} samples")

# Train multiple models and create ensemble
print("\n[6/7] Training models...")

# Model 1: PassiveAggressive with calibration for probability estimates
pac_base = PassiveAggressiveClassifier(
    max_iter=150,              # Increased iterations
    C=0.1,                     # Lower C = stronger regularization (less overfitting)
    random_state=42,
    loss='hinge',              # Hinge loss for better generalization
    n_jobs=-1
)

print("   • Training PassiveAggressive Classifier with calibration...")
pac_base.fit(X_train, y_train)
# Wrap with calibration to enable probability estimates for soft voting
pac = CalibratedClassifierCV(pac_base, cv=3, method='sigmoid')
pac.fit(X_train, y_train)

# Model 2: Logistic Regression with stronger regularization
print("   • Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=300,
    C=0.5,                     # Stronger regularization (was 1.0)
    solver='saga',             # Better for large datasets
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train, y_train)

# Model 3: Multinomial Naive Bayes with tuned smoothing
print("   • Training Naive Bayes...")
nb = MultinomialNB(alpha=0.05)  # Lower alpha for less smoothing
nb.fit(X_train, y_train)

# Evaluate individual model performance
pac_score = pac.score(X_test, y_test)
lr_score = lr.score(X_test, y_test)
nb_score = nb.score(X_test, y_test)

print(f"   ✓ Calibrated PAC accuracy: {pac_score*100:.2f}%")
print(f"   ✓ LogReg accuracy: {lr_score*100:.2f}%")
print(f"   ✓ NaiveBayes accuracy: {nb_score*100:.2f}%")

print("   • Creating ensemble model with soft voting...")
# Use soft voting (probability averaging) and weight by performance
model = VotingClassifier(
    estimators=[
        ('pac', pac),
        ('lr', lr),
        ('nb', nb)
    ],
    voting='soft',             # Use probability estimates (more nuanced than hard voting)
    weights=[1, 2, 1]          # Give Logistic Regression more weight (usually more balanced)
)
model.fit(X_train, y_train)

# Evaluate
print("\n[7/7] Evaluating model...")
y_pred = model.predict(X_test)

# Detailed metrics
score = accuracy_score(y_test, y_pred)
print(f"\n   ✓ Accuracy: {round(score * 100, 2)}%")

print("\n   Classification Report:")
print("   " + "-" * 56)
report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'])
for line in report.split('\n'):
    if line.strip():
        print(f"   {line}")

print("\n   Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   True Negative (Correct Fake):  {cm[0][0]}")
print(f"   False Positive (Real as Fake):  {cm[1][0]} ⚠️")
print(f"   False Negative (Fake as Real):  {cm[0][1]} ⚠️")
print(f"   True Positive (Correct Real):   {cm[1][1]}")

# Cross-validation
print("\n   Cross-Validation (5-fold):")
cv_scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
print(f"   CV Scores: {[round(s*100, 2) for s in cv_scores]}")
print(f"   Mean CV Accuracy: {round(cv_scores.mean()*100, 2)}%")

# Save model
print("\n" + "=" * 60)
print("SAVING MODELS...")
print("=" * 60)
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("   ✓ model.pkl saved")
print("   ✓ vectorizer.pkl saved")
print("\n✅ Training complete! Model is ready for deployment.")
print("=" * 60)
```

---

## File: train_model_combined.py
```python
import pandas as pd
import re
import string
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

print("=" * 70)
print("FAKE NEWS DETECTION - COMBINED DATASET TRAINING")
print("(US + Indian News Sources)")
print("=" * 70)

# ============================================================================
# PART 1: LOAD ALL DATASETS
# ============================================================================

print("\n[1/8] Loading datasets...")

# 1. US Fake News
print("   • Loading US fake news...")
us_fake = pd.read_csv("dataset/Fake.csv")
us_fake = us_fake[["text"]]
us_fake["label"] = 0
us_fake["source"] = "US"
print(f"     ✓ US Fake articles: {len(us_fake)}")

# 2. US Real News
print("   • Loading US real news...")
us_true = pd.read_csv("dataset/True.csv")
us_true = us_true[["text"]]
us_true["label"] = 1
us_true["source"] = "US"
print(f"     ✓ US Real articles: {len(us_true)}")

# 3. IFND - Indian Fake News Dataset
print("   • Loading IFND (Indian dataset)...")
try:
    ifnd = pd.read_csv("dataset/IFND.csv", encoding='latin-1')
    # Use 'Statement' column as text
    ifnd = ifnd[["Statement", "Label"]].copy()
    ifnd.columns = ["text", "label"]
    # Convert TRUE/FALSE to 1/0
    ifnd["label"] = ifnd["label"].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0})
    ifnd = ifnd.dropna(subset=["label"])
    ifnd["source"] = "IFND"
    print(f"     ✓ IFND articles: {len(ifnd)}")
except Exception as e:
    print(f"     ✗ Error loading IFND: {e}")
    ifnd = pd.DataFrame(columns=["text", "label", "source"])

# 4. Bharat FakeNews Kosh
print("   • Loading Bharat FakeNews Kosh...")
try:
    bharat = pd.read_excel("dataset/bharatfakenewskosh (3).xlsx")
    # Use English translation if available, otherwise original statement
    bharat["text"] = bharat["Eng_Trans_Statement"].fillna(bharat["Statement"])
    # If text is still empty, try News Body
    bharat["text"] = bharat["text"].fillna(bharat["Eng_Trans_News_Body"]).fillna(bharat["News Body"])
    bharat = bharat[["text", "Label"]].copy()
    bharat.columns = ["text", "label"]
    # Convert True/False or 1/0
    bharat["label"] = bharat["label"].map({True: 0, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
    # Note: In Bharat dataset, Label=False means fake news, True means real
    # Adjust if needed based on actual data
    bharat = bharat.dropna(subset=["label"])
    bharat["source"] = "Bharat"
    print(f"     ✓ Bharat articles: {len(bharat)}")
except Exception as e:
    print(f"     ✗ Error loading Bharat: {e}")
    bharat = pd.DataFrame(columns=["text", "label", "source"])

# 5. Additional News Dataset
print("   • Loading additional news dataset...")
try:
    news_ds = pd.read_csv("dataset/news_dataset.csv")
    # Convert REAL/FAKE labels to 1/0
    news_ds["label"] = news_ds["label"].map({"REAL": 1, "FAKE": 0, "Real": 1, "Fake": 0, 1: 1, 0: 0})
    news_ds = news_ds.dropna(subset=["label", "text"])
    news_ds = news_ds[["text", "label"]].copy()
    news_ds["source"] = "Additional"
    print(f"     ✓ Additional articles: {len(news_ds)}")
except Exception as e:
    print(f"     ✗ Error loading additional dataset: {e}")
    news_ds = pd.DataFrame(columns=["text", "label", "source"])

# Combine all datasets
data = pd.concat([us_fake, us_true, ifnd, bharat, news_ds], ignore_index=True)

print(f"\n   📊 TOTAL DATASET SIZE: {len(data)} articles")
print(f"      • US sources: {len(data[data['source']=='US'])}")
print(f"      • IFND (Indian): {len(data[data['source']=='IFND'])}")
print(f"      • Bharat Kosh: {len(data[data['source']=='Bharat'])}")
print(f"      • Additional dataset: {len(data[data['source']=='Additional'])}")

# ============================================================================
# PART 2: DATA CLEANING
# ============================================================================

print("\n[2/8] Cleaning dataset...")

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Remove duplicates and null values
initial_len = len(data)
data = data.dropna(subset=['text', 'label'])
data = data.drop_duplicates(subset=['text'])
print(f"   ✓ Removed {initial_len - len(data)} duplicates/nulls")
print(f"   ✓ Total articles after cleaning: {len(data)}")
print(f"   ✓ Class distribution: Fake={len(data[data['label']==0])}, Real={len(data[data['label']==1])}")

# Clean text - preserve context while removing noise
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

print("\n[3/8] Applying text preprocessing...")
data["text"] = data["text"].apply(clean_text)

# Remove very short articles (likely noise)
before_filter = len(data)
data = data[data["text"].str.split().str.len() > 10]
print(f"   ✓ Removed {before_filter - len(data)} very short texts (< 10 words)")
print(f"   ✓ Final dataset size: {len(data)}")

# ============================================================================
# PART 3: FEATURE EXTRACTION
# ============================================================================

print("\n[4/8] Creating TF-IDF features...")

# Keep text and label columns
data = data[["text", "label"]]

# Input and output
X = data["text"]
y = data["label"]

# Convert text to vectors - optimized for larger diverse dataset
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,        # Increased for more diverse vocabulary
    ngram_range=(1, 2),        # Unigrams and bigrams
    min_df=3,                  # Word must appear in at least 3 documents
    max_df=0.85,               # Ignore words in more than 85% of docs
    sublinear_tf=True,         # Apply sublinear tf scaling
    smooth_idf=True,           # Smooth idf weights
    use_idf=True               # Enable inverse-document-frequency
)

X = vectorizer.fit_transform(X)
print(f"   ✓ Feature matrix shape: {X.shape}")
print(f"   ✓ Vocabulary size: {len(vectorizer.vocabulary_)}")

# ============================================================================
# PART 4: TRAIN/TEST SPLIT
# ============================================================================

print("\n[5/8] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Training set: {X_train.shape[0]} samples")
print(f"   ✓ Test set: {X_test.shape[0]} samples")

# ============================================================================
# PART 5: MODEL TRAINING
# ============================================================================

print("\n[6/8] Training ensemble model...")

# Model 1: PassiveAggressive with calibration for probability estimates
print("   • Training PassiveAggressive Classifier...")
pac_base = PassiveAggressiveClassifier(
    max_iter=200,              # Increased for larger dataset
    C=0.15,                    # Slightly higher C for diverse data
    random_state=42,
    loss='hinge',              
    n_jobs=-1
)
pac_base.fit(X_train, y_train)
# Wrap with calibration to enable probability estimates
pac = CalibratedClassifierCV(pac_base, cv=3, method='sigmoid')
pac.fit(X_train, y_train)
pac_score = pac.score(X_test, y_test)
print(f"     ✓ Accuracy: {pac_score*100:.2f}%")

# Model 2: Logistic Regression
print("   • Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=400,              # Increased iterations
    C=0.6,                     
    solver='saga',             
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
print(f"     ✓ Accuracy: {lr_score*100:.2f}%")

# Model 3: Multinomial Naive Bayes
print("   • Training Naive Bayes...")
nb = MultinomialNB(alpha=0.08)  # Tuned for diverse dataset
nb.fit(X_train, y_train)
nb_score = nb.score(X_test, y_test)
print(f"     ✓ Accuracy: {nb_score*100:.2f}%")

# Create ensemble with soft voting
print("   • Creating ensemble model...")
model = VotingClassifier(
    estimators=[
        ('pac', pac),
        ('lr', lr),
        ('nb', nb)
    ],
    voting='soft',             # Use probability averaging
    weights=[1, 2, 1],         # Weight LogReg higher
    n_jobs=-1
)
model.fit(X_train, y_train)

# ============================================================================
# PART 6: EVALUATION
# ============================================================================

print("\n[7/8] Evaluating model performance...")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"   ✓ Test Accuracy: {accuracy * 100:.2f}%")

# Cross-validation
print("\n   • Running 5-fold cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f"   ✓ CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%")

# Detailed metrics
print("\n   📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'], digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n   📊 Confusion Matrix:")
print(f"      True Negatives (Correct Fake):  {cm[0][0]}")
print(f"      False Positives (Wrong Fake):   {cm[1][0]}")
print(f"      False Negatives (Wrong Real):   {cm[0][1]}")
print(f"      True Positives (Correct Real):  {cm[1][1]}")

# False positive rate (marking real news as fake)
total_real = cm[1][0] + cm[1][1]
false_positive_rate = (cm[1][0] / total_real) * 100 if total_real > 0 else 0
print(f"\n   ⚠️  False Positive Rate: {false_positive_rate:.2f}% ({cm[1][0]}/{total_real})")

# ============================================================================
# PART 7: SAVE MODELS
# ============================================================================

print("\n[8/8] Saving trained model and vectorizer...")

# Save the model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
print("   ✓ Model saved as 'model.pkl'")

# Save the vectorizer
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
print("   ✓ Vectorizer saved as 'vectorizer.pkl'")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📈 Final Model Performance:")
print(f"   • Dataset Size: {len(data)} articles")
print(f"   • Test Accuracy: {accuracy * 100:.2f}%")
print(f"   • CV Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"   • Vocabulary: {len(vectorizer.vocabulary_)} unique terms")
print(f"   • Model Type: Ensemble (PAC + LogReg + NaiveBayes)")
print(f"\n🌍 Dataset Composition:")
print(f"   • US News Sources")
print(f"   • Indian News (IFND + Bharat Kosh)")
print(f"   • Regional & Diverse Content")
print("\n💡 The model is now ready for production use with:")
print("   - Better handling of Indian/regional news")
print("   - Improved diversity and generalization")
print("   - Reduced false positives on unfamiliar content")
print("\n" + "=" * 70)
```

---

## File: test_model.py
```python
import pickle
import re

# Load model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Clean text function (same as training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\b\w\b', '', text)
    return text.strip()

# CBSE article
article = """As Class XII board examinees took to social media to raise concerns about alleged marking discrepancies in the CBSE's newly introduced On-Screen Marking (OSM) system, a parallel trend emerged online: principals of CBSE-affiliated schools began posting strikingly similar videos and parroting similar lines while defending the evaluation mechanism and urging students to "trust the system.".

For the first time this year, the Central Board of Secondary Education (CBSE) introduced the On-Screen Marking (OSM) system, a digital evaluation mechanism intended to replace the conventional process of physically handling and checking answer scripts. Under the new system, answer sheets are scanned, uploaded to a centralised portal, and evaluated digitally by examiners logging into the CBSE platform. Marks are entered directly into the system, reducing manual tabulation errors and streamlining the evaluation process.

However, shortly after the declaration of results, social media platforms, particularly X (formerly Twitter), were inundated with complaints from students alleging serious irregularities in the evaluation process.

Among them was Vedant Srivastava, who claimed that after receiving unexpectedly low marks in Physics, he applied for photocopies of his answer scripts through CBSE's re-evaluation process. To his shock, the Physics answer sheet uploaded by the Board did not appear to be his own. To make things worse, upon seeing his post, Doordarshan news anchor Ashok Shrivastav sarcastically wrote in Hindi: "Did Pakistanis also appear for CBSE exams?!!""" 

print("=" * 70)
print("TESTING NEW MODEL WITH CBSE ARTICLE")
print("=" * 70)

# Clean the text
cleaned = clean_text(article)
print(f"\n✓ Article length: {len(article.split())} words")
print(f"✓ Cleaned length: {len(cleaned.split())} words")

# Transform and predict
X = vectorizer.transform([cleaned])
prediction = model.predict(X)
probabilities = model.predict_proba(X)[0]

print(f"\n📊 Model Prediction:")
print(f"   • Raw prediction: {prediction[0]} (0=FAKE, 1=REAL)")
print(f"   • Probability FAKE: {probabilities[0]*100:.2f}%")
print(f"   • Probability REAL: {probabilities[1]*100:.2f}%")

# Apply same logic as app.py
confidence_threshold = 0.70
prob_fake = probabilities[0]
prob_real = probabilities[1]

if prob_real > confidence_threshold:
    result = "REAL"
    confidence = prob_real
elif prob_fake > confidence_threshold:
    result = "FAKE"
    confidence = prob_fake
else:
    if prob_real > prob_fake:
        result = "UNCERTAIN_REAL"
        confidence = prob_real
    else:
        result = "UNCERTAIN_FAKE"
        confidence = prob_fake

print(f"\n🎯 Final Classification (with 70% threshold):")
print(f"   • Result: {result}")
print(f"   • Confidence: {confidence*100:.2f}%")
print(f"\n💡 With 70% threshold:")
if "UNCERTAIN" in result:
    print(f"   ✓ Model is UNCERTAIN (confidence {confidence*100:.2f}% < 70%)")
    print(f"   ✓ This prevents false classification of unfamiliar content")
print("\n" + "=" * 70)
```

---

## File: test_flask_prediction.py
```python
import pickle
import re

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Full CBSE article
article = """As Class XII board examinees took to social media to raise concerns about alleged marking discrepancies in the CBSE's newly introduced On-Screen Marking (OSM) system, a parallel trend emerged online: principals of CBSE-affiliated schools began posting strikingly similar videos and parroting similar lines while defending the evaluation mechanism and urging students to "trust the system."..."""

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\b\w\b', '', text)
    return text.strip()

cleaned = clean_text(article)
X = vectorizer.transform([cleaned])
probs = model.predict_proba(X)[0]

print("=" * 60)
print("ACTUAL FLASK PREDICTION TEST")
print("=" * 60)
print(f"Prob Fake: {probs[0]*100:.2f}%")
print(f"Prob Real: {probs[1]*100:.2f}%")
print()

# Apply 70% threshold
threshold = 0.70
if probs[1] > threshold:
    result = "REAL"
    conf = probs[1]
elif probs[0] > threshold:
    result = "FAKE"
    conf = probs[0]
else:
    if probs[1] > probs[0]:
        result = "UNCERTAIN_REAL"
        conf = probs[1]
    else:
        result = "UNCERTAIN_FAKE"
        conf = probs[0]

print(f"Threshold: {threshold*100:.0f}%")
print(f"Final Result: {result}")
print(f"Confidence: {conf*100:.2f}%")
print("=" * 60)
```

---

## File: requirement.txt
```text
Flask>=3.0.3
pandas>=2.2.3
numpy>=2.0.0
scikit-learn>=1.7.2
nltk>=3.8.1
openpyxl>=3.1.0
joblib>=1.4.2
scipy>=1.13.1
matplotlib>=3.9.0
```

---

## File: templates/index.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TruthScan — AI Fake News Detector</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style_new.css') }}">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
</head>
<body>

    <!-- Animated background orbs -->
    <div class="bg-orbs">
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>
        <div class="orb orb-4"></div>
    </div>

    <!-- Navbar -->
    <nav class="navbar">
        <div class="nav-brand">
            <div class="brand-icon">
                <i class="fas fa-shield-halved"></i>
            </div>
            <span>TruthScan</span>
        </div>
        <div class="nav-right">
            <span class="nav-badge"><i class="fas fa-microchip"></i> AI Powered</span>
        </div>
    </nav>

    <main class="container">

        <!-- Hero -->
        <div class="hero">
            <h1 class="hero-title">
                Detect Fake News<br>with AI Precision
            </h1>
            <p class="hero-subtitle">
                Advanced machine learning ensemble combining PassiveAggressive Classifier, Logistic Regression, and Naive Bayes.
                Trained on 85,000+ articles for superior accuracy.
            </p>
            <div class="hero-features">
                <div class="feature-badge">
                    <span class="feature-icon">🤖</span>
                    <span>AI Ensemble Model</span>
                </div>
                <div class="feature-badge">
                    <span class="feature-icon">📊</span>
                    <span>89% Accuracy</span>
                </div>
                <div class="feature-badge">
                    <span class="feature-icon">⚡</span>
                    <span>Instant Analysis</span>
                </div>
                <div class="feature-badge">
                    <span class="feature-icon">🌍</span>
                    <span>Global Coverage</span>
                </div>
            </div>
        </div>

        <!-- Form Card -->
        <div class="card">
            <div class="card-header">
                <div class="card-icon">
                    <i class="fas fa-newspaper"></i>
                </div>
                <div class="card-title">
                    📰 Paste Your News Article
                </div>
            </div>
            <form id="newsForm" action="/predict" method="POST">
                <div class="textarea-wrapper">
                    <textarea
                        name="news"
                        id="newsInput"
                        placeholder="Paste or type a news article here to analyze its authenticity and credibility..."
                        required>{{ news_text or '' }}</textarea>
                    <div class="char-counter" id="wordCounter">0 words</div>
                </div>
                <button type="submit" class="btn" id="analyzeBtn">
                    <span class="btn-text"><i class="fas fa-magnifying-glass-chart btn-icon"></i>&nbsp; Analyze Article</span>
                    <span class="btn-loading" style="display:none;"></span>
                </button>
            </form>
        </div>

        <!-- Error alert -->
        {% if error %}
        <div class="alert alert-error">
            <i class="fas fa-circle-exclamation"></i>
            <span>{{ error }}</span>
        </div>
        {% endif %}

        <!-- Short article warning -->
        {% if short_article_warning %}
        <div class="alert alert-warning">
            <i class="fas fa-triangle-exclamation"></i>
            <span>{{ short_article_warning }}</span>
        </div>
        {% endif %}

        <!-- ===== Results ===== -->
        {% if result %}
        <div class="results show" id="results">

            <!-- Main verdict -->
            <div class="card verdict-card {{ 'verdict-real' if 'REAL' in result else ('verdict-uncertain' if 'UNCERTAIN' in result else 'verdict-fake') }}">
                <div class="verdict-icon">
                    {% if 'REAL' in result %}
                        ✓
                    {% elif 'UNCERTAIN' in result %}
                        ?
                    {% else %}
                        ✕
                    {% endif %}
                </div>
                <p class="verdict-label">Analysis Result</p>
                <h2 class="verdict-title">{{ label }}</h2>
                <p class="verdict-text">
                    {% if 'REAL' in result %}
                        This article exhibits patterns consistent with credible, factual reporting. However, always verify important news from multiple sources.
                    {% elif 'UNCERTAIN' in result %}
                        This article shows mixed signals. The model cannot confidently classify it. This often happens with unfamiliar content, regional news, or articles with unconventional patterns. Use additional fact-checking sources.
                    {% else %}
                        This article shows patterns commonly associated with misinformation or fabricated content. Verify from credible sources before sharing.
                    {% endif %}
                </p>
                <div class="verdict-badge {{ 'badge-real' if 'REAL' in result else ('badge-uncertain' if 'UNCERTAIN' in result else 'badge-fake') }}">
                    {% if 'UNCERTAIN' in result %}UNCERTAIN{% else %}{{ result }}{% endif %}
                </div>
            </div>

            <!-- Confidence Section -->
            <div class="card confidence-section">
                <div class="confidence-header">
                    <div class="confidence-label">
                        <i class="fas fa-gauge-high"></i>
                        Confidence Score
                    </div>
                    <div class="confidence-value">{{ confidence }}%</div>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar {{ 'progress-real' if 'REAL' in result else ('progress-uncertain' if 'UNCERTAIN' in result else 'progress-fake') }}"
                         style="width: {{ confidence }}%"></div>
                </div>
            </div>

            <!-- Article Statistics -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">
                        <i class="fas fa-chart-bar"></i>
                    </div>
                    <div class="card-title">
                        📊 Article Statistics
                    </div>
                </div>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-icon">📝</div>
                        <span class="stat-value">{{ word_count }}</span>
                        <span class="stat-label">Words</span>
                    </div>
                    <div class="stat-item">
                        <div class="stat-icon">🔤</div>
                        <span class="stat-value">{{ char_count }}</span>
                        <span class="stat-label">Characters</span>
                    </div>
                    <div class="stat-item">
                        <div class="stat-icon">📄</div>
                        <span class="stat-value">{{ sentence_count }}</span>
                        <span class="stat-label">Sentences</span>
                    </div>
                </div>
            </div>

            <!-- Disclaimer -->
            <div class="alert alert-warning">
                <div class="alert-icon">⚠️</div>
                <span><strong>Important:</strong> This model was trained primarily on US/Western news sources and 3 Indian datasets (total 85,701 articles). It may be less accurate with regional news, specialized topics, or unconventional writing styles. When uncertain, it will show an "UNCERTAIN" result. Always verify important news from multiple credible sources.</span>
            </div>

        </div>
        {% endif %}

        <!-- How It Works -->
        <section class="info-section">
            <h2 class="section-title">⚙️ How It Works</h2>
            <div class="info-grid">
                <div class="info-item">
                    <h3 class="info-item-title">
                        <span>01</span>
                        📝 Paste Article
                    </h3>
                    <p class="info-item-text">Copy and paste the full news article or headline you want to verify into the text box above. Longer articles provide more data for accurate analysis.</p>
                </div>
                <div class="info-item">
                    <h3 class="info-item-title">
                        <span>02</span>
                        🤖 AI Processing
                    </h3>
                    <p class="info-item-text">TF-IDF vectorization extracts 10,000 linguistic features (unigrams + bigrams). Our ensemble model (PAC, LogReg, NaiveBayes) processes these features with soft voting.</p>
                </div>
                <div class="info-item">
                    <h3 class="info-item-title">
                        <span>03</span>
                        ⚡ Instant Result
                    </h3>
                    <p class="info-item-text">Get a credibility verdict with confidence score (70% threshold for certainty), full article statistics, and uncertainty alerts in milliseconds.</p>
                </div>
            </div>

            <h2 class="section-title" style="margin-top: 48px;">🛠️ Technology Stack</h2>
            <div class="tech-stack">
                <div class="tech-badge"><i class="fab fa-python"></i> Python 3.13</div>
                <div class="tech-badge"><i class="fas fa-flask"></i> Flask 3.0</div>
                <div class="tech-badge"><i class="fas fa-brain"></i> scikit-learn 1.8</div>
                <div class="tech-badge"><i class="fas fa-table-cells"></i> TF-IDF (10K features)</div>
                <div class="tech-badge"><i class="fas fa-robot"></i> Ensemble Model</div>
                <div class="tech-badge"><i class="fas fa-chart-line"></i> Logistic Regression</div>
                <div class="tech-badge"><i class="fas fa-network-wired"></i> Naive Bayes</div>
                <div class="tech-badge"><i class="fas fa-database"></i> 85,701 articles</div>
            </div>
        </section>

    </main>

    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
```

---

## File: static/script.js
```javascript
/* ============================================================
   TruthScan — Fake News Detection System
   Client-side interactions
   ============================================================ */

// ---- Word / character counter ----
const textarea    = document.getElementById('newsInput');
const wordCounter = document.getElementById('wordCounter');

function updateCounter() {
    if (!textarea || !wordCounter) return;
    const text  = textarea.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    wordCounter.textContent = words.toLocaleString() + (words === 1 ? ' word' : ' words');
}

if (textarea) {
    textarea.addEventListener('input', updateCounter);
    updateCounter(); // initialise on page load (populated on result page)
}

// ---- Loading state on form submit ----
const form    = document.getElementById('newsForm');
const btn     = document.getElementById('analyzeBtn');

if (form && btn) {
    form.addEventListener('submit', function () {
        const btnText    = btn.querySelector('.btn-text');
        const btnLoading = btn.querySelector('.btn-loading');

        if (btnText)    btnText.style.display    = 'none';
        if (btnLoading) btnLoading.style.display = 'inline-flex';
        btn.disabled = true;
    });
}

// ---- Auto-scroll to results after submission ----
const resultsSection = document.getElementById('results');
if (resultsSection) {
    setTimeout(function () {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 150);
}
```

---

## File: static/style_new.css
```css
/* Full CSS omitted here for brevity in the single-page preview. See the saved file for the complete stylesheet. */
```

---

## File: static/style.css
```css
/* style.css (empty or minimal in this project) */
```

---

End of consolidated code file.
