import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("HGBC.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("predict.html")

@app.route("/predict", methods = ["POST"])
def predict():
    noi = request.form.get('noi')
    soi = request.form.get('soi')
    rpi = request.form.get('rpi')
    spmi = request.form.get('spmi')

    input_query = np.array([[noi, soi, rpi, spmi]])

    result = model.predict(input_query)[0]

    return jsonify({'result': result})


if __name__ == "__main__":
    app.run(debug=True)
