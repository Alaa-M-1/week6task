from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('lr_model.pkl')
vectorizer = joblib.load('vect.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_text = data['body']
    input_vector = vectorizer.transform([input_text])
    prediction = model.predict(input_vector)
    if prediction ==0:
        return jsonify({'prediction': "The review is negative."})
    else:
        return jsonify({'prediction': "The review is positive."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT from environment or default to 5000
    app.run(host='0.0.0.0', port=port)
