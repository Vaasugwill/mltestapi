from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
poly_reg = joblib.load('polynomial_features.pkl')
pol_reg = joblib.load('polynomial_regression.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    user_input = np.array(data['input']).reshape(-1, 1)
    X_new_poly = poly_reg.transform(user_input)
    predictions = pol_reg.predict(X_new_poly)
    return jsonify({'prediction':round(predictions[0][0],1)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
