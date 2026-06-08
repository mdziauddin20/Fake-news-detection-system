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