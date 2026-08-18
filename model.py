from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
texts=['Government launches new policy','Scientists discover planet','Aliens rule earth','Magic pill grants immortality']
labels=[1,1,0,0]
vec=TfidfVectorizer()
X=vec.fit_transform(texts)
model=MultinomialNB().fit(X,labels)
def predict_news(text):
    pred=model.predict(vec.transform([text]))[0]
    return 'Real News' if pred==1 else 'Fake News'
