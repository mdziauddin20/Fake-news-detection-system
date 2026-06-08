import pickle
import re

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Full CBSE article
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
