from flask import Flask, render_template, jsonify
from my_prediction_module import perform_prediction

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def predict():
    # Perform prediction using the Python code
    prediction = perform_prediction()

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run()
