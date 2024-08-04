from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('lr_model.pkl')
vectorizer = joblib.load('vect.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_text = data['body']
    input_vector = vectorizer.transform([input_text])
    prediction = model.predict(input_vector)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
