# app.py
from flask import Flask, request, render_template
import pickle
import numpy as np

# Load trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_data = request.form['features']
        input_features = [float(x.strip()) for x in raw_data.split(',')]

        if len(input_features) != 31:
            return render_template('index.html', prediction_text="⚠️ Please enter exactly 31 features.")

        input_array = np.array(input_features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        prediction = model.predict(scaled_input)[0]
        result = '🛑 Cancerous' if prediction == 1 else '✅ Not Cancerous'

        # Additional insights
        radius_mean = input_features[0]  # Feature 1
        symmetry_mean = input_features[12]  # Feature 13

        if radius_mean < 12:
            radius_category = "Small"
        elif radius_mean < 18:
            radius_category = "Medium"
        else:
            radius_category = "Large"

        if symmetry_mean < 0.18:
            symmetry_level = "Low"
        elif symmetry_mean < 0.25:
            symmetry_level = "Medium"
        else:
            symmetry_level = "High"

        extra_info = f"Tumor Radius: {radius_category} | Symmetry: {symmetry_level}"
        return render_template('index.html', prediction_text=result + "<br>" + extra_info)

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)