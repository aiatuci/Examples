from flask import Flask, render_template, request
from Predictor import Predictor

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get('review')
    print(review)
    isPositive = app.predictor.predict(review)
    return render_template("canny.html") if isPositive else render_template("uncanny.html")

def start_app():
    app.predictor = Predictor()
    app.run()

if __name__ == "__main__":
    start_app()