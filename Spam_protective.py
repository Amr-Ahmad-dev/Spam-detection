import re
import nltk
import pandas as pd
import joblib
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords", quiet=True)

url = r"C:\Users\amrrrr\Downloads\spam_4\spam (4).csv"
df = pd.read_csv(url, encoding="latin-1")


stop = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
token_re = re.compile(r"[a-z]+")          

def clean(text: str) -> str:
    tokens = token_re.findall(text.lower())
    return " ".join(stemmer.stem(t) for t in tokens if t not in stop)

df["clean"] = df["v2"].apply(clean)  
df[["v1", "clean"]].to_csv(r"C:\Users\amrrrr\Downloads\spam_4\spam_cleaned.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["v1"],
    test_size=0.2, random_state=42, stratify=df["v1"],
)

model = Pipeline([
    ("vec", CountVectorizer()),           
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)

score =model.score(X_test, y_test)
print("Model Score:", score)

print(classification_report(y_test, pred))

joblib.dump(model, "spam_model.pkl")       