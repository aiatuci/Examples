from flask import Flask, render_template, request
from Predictor import Predictor

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get('review')
    rating = app.predictor.predict(review)
    return render_template("predict.html", rating=rating)

def start_app():
    app.predictor = Predictor()
    app.run()

if __name__ == "__main__":
    start_app()