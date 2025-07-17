from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
app = Flask(__name__)

print("[Debug] Loading Model")
model = joblib.load('gwp.pkl')
print("[DEBUG] Model loaded. Features used in training:", model.feature_names_in_)
print("[Debug] Model Loaded Successfully")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    print("[Debug] Reached predict route")
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        print("[Debug] Reached submit route")  # Debugging line to check if the route is hit
        print("[Debug] Request Form Data:", request.form)  # Debugging line to check
        try:
            day = int(request.form['day'])
            team = int(request.form['team'])
            targeted = float(request.form['targeted_productivity'])
            smv = float(request.form['smv'])
            overtime = float(request.form['over_time'])  # name matches 'over_time'
            incentive = float(request.form['incentive'])
            workers = int(request.form['no_of_workers'])
            workday = int(request.form['workday'])  # 0 or 1
            smv_worker = smv * workers

            columns = ['day', 'team', 'targeted_productivity', 'smv', 'over_time',
                       'incentive', 'no_of_workers', 'workday', 'smv_worker']

            input_df = pd.DataFrame([[day, team, targeted, smv, overtime, incentive, workers, workday, smv_worker]],
                                columns=columns)
            print(f"Input DataFrame: {input_df}")  # Debugging line to check input DataFrame
            prediction = model.predict(input_df)[0]
            print(f"Prediction: {prediction}")  # Debugging line to check prediction value
            return render_template('submit.html', predicted_text=f'{round(prediction, 2)}')
        except Exception as e:
            return render_template('submit.html', predicted_text=f'Error: {str(e)}')
        


if __name__ == '__main__':
    app.run(debug=True)
