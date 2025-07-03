from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('models/churn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json['features']
    prediction = model.predict([np.array(features)])
    return jsonify({'churn_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
