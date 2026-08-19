import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

print("1. Loading dataset...")
df = pd.read_csv("WELFake_Dataset.csv")

# 2. Clean data
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

# Combine and lowercase
df["content"] = (df["title"] + " " + df["text"]).str.lower()

X = df["content"]
y = df["label"]  # 0 = Fake, 1 = Real

# Stratified split ensures balanced class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("3. Vectorizing...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.8,
    min_df=2,
    max_features=40000,
    ngram_range=(1, 2),
    sublinear_tf=True,
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("4. Training Logistic Regression model...")
# class_weight='balanced' ensures neither Fake nor Real dominates predictions
model = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
model.fit(X_train_vec, y_train)

print("\nModel Performance:")
print(classification_report(y_test, model.predict(X_test_vec)))

# 5. Save artifacts
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Training finished successfully!")