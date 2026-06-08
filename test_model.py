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
article = """As Class XII board examinees took to social media to raise concerns about alleged marking discrepancies in the CBSE's newly introduced On-Screen Marking (OSM) system, a parallel trend emerged online: principals of CBSE-affiliated schools began posting strikingly similar videos and parroting similar lines while defending the evaluation mechanism and urging students to "trust the system."

For the first time this year, the Central Board of Secondary Education (CBSE) introduced the On-Screen Marking (OSM) system, a digital evaluation mechanism intended to replace the conventional process of physically handling and checking answer scripts. Under the new system, answer sheets are scanned, uploaded to a centralised portal, and evaluated digitally by examiners logging into the CBSE platform. Marks are entered directly into the system, reducing manual tabulation errors and streamlining the evaluation process.

However, shortly after the declaration of results, social media platforms, particularly X (formerly Twitter), were inundated with complaints from students alleging serious irregularities in the evaluation process.

Among them was Vedant Srivastava, who claimed that after receiving unexpectedly low marks in Physics, he applied for photocopies of his answer scripts through CBSE's re-evaluation process. To his shock, the Physics answer sheet uploaded by the Board did not appear to be his own. To make things worse, upon seeing his post, Doordarshan news anchor Ashok Shrivastav sarcastically wrote in Hindi: "Did Pakistanis also appear for CBSE exams?!!"
Dozens of students have since alleged that the scanned copies of their answer scripts were blurry, partially missing, displayed blank pages, or were otherwise difficult to read. Several students also claimed that correct MCQ responses were awarded only partial marks, step marking had been ignored, answers appeared unchecked, and overall scores were inexplicably lower than expected.
Amid mounting outrage and confusion, another curious pattern began to emerge online.

A deluge of videos featuring principals, faculty members, and administrators from CBSE-affiliated schools suddenly appeared across various social media platforms, many uploaded through official school accounts. This seemed like a strange coincidence. Across these videos, school administrators repeatedly described the transition from traditional paper-checking to digital evaluation as a "monumental step" towards "modernising" the examination ecosystem. 

The language, tone, and structure of these statements bore uncanny similarities, with many institutions seemingly parroting near-identical scripts defending the OSM system and urging students to "trust the process." 

One such post was uploaded by Delhi Public School (DPS), Nerul, Navi Mumbai, on its official Facebook page in response to growing anxieties surrounding the OSM rollout. The school described the shift from manual checking to digital evaluation as a "massive step towards modernizing the examination ecosystem" and characterised the transition as a "monumental reform" aimed at ensuring "fundamental improvements" in the evaluation process.

The post highlighted the purported benefits of OSM, including automated total calculations and question-wise score mapping, which it claimed would eliminate clerical errors such as calculation slips and incorrect posting of marks that have historically plagued manual evaluation systems.

Addressing the growing concerns raised by students regarding discrepancies in their digital answer scripts, the school urged them not to panic. Acknowledging the unprecedented scale of the exercise — "around 98 lakh answer sheets were digitized globally" — the statement conceded that "implementation bumps" were inevitable but assured students that no one would be disadvantaged due to technical errors.

The post further asserted that the CBSE had been "highly proactive," "empathetic," and "communicative" in addressing complaints. It concluded by urging students and parents to "embrace these digital advancements with patience" and "trust the system."

One notable detail, however, stood out: instead of being signed off by the school principal, the statement was attributed to the "City Coordinator CBSE."
The post has currently been deleted from the page.

The videos followed a similarly familiar template. Principals opened by acknowledging students' "questions and concerns" surrounding the newly introduced evaluation mechanism before describing the OSM rollout as a "crucial milestone" in the evolution of the examination process.

Alt News found striking similarities between the DPS Nerul statement and videos uploaded by principals from several schools, many of whom appeared to be reading from identical or near-identical scripts."""

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
