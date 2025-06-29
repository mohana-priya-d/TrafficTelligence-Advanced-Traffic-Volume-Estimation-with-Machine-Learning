from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('traffic_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        holiday = int(request.form['holiday'])
        weather = int(request.form['weather'])
        hour = int(request.form['hour'])
        dayofweek = int(request.form['dayofweek'])

        features = np.array([[holiday, temp, rain, snow, weather, hour, dayofweek]])
        prediction = model.predict(features)
        predicted_volume = int(prediction[0])

        return render_template('index.html', prediction_text=f'🔮 Predicted Traffic Volume: {predicted_volume}')
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)