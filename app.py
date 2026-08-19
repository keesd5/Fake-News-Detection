import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        news = request.form.get("news", "")
        if news.strip():
            # Clean and vectorize
            clean_text = news.lower().strip()
            vec = vectorizer.transform([clean_text])

            # Get probabilities [class_0, class_1]
            probs = model.predict_proba(vec)[0]
            prob_0 = probs[0]  # Real News in your dataset
            prob_1 = probs[1]  # Fake News in your dataset

            # Flipped mapping to match your dataset labels:
            if prob_1 >= 0.50:
                result = "Fake News"
                confidence = f"{prob_1 * 100:.1f}% Confidence"
            else:
                result = "Real News"
                confidence = f"{prob_0 * 100:.1f}% Confidence"

    return render_template("index.html", result=result, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)