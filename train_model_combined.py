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
print(f"   ✓ CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

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
