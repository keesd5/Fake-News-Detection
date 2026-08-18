from flask import Flask, render_template, request
from model import predict_news
app = Flask(__name__)
@app.route('/', methods=['GET','POST'])
def home():
    result=None
    if request.method=='POST':
        result=predict_news(request.form['news'])
    return render_template('index.html', result=result)
if __name__=='__main__':
    app.run(debug=True)
