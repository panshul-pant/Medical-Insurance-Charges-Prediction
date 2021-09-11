from flask import Flask
from flask import url_for, redirect, render_template, request


import numpy as np, pandas as pd
import joblib

with open('xgb_model.joblib', 'rb') as f:
    model = joblib.load(f)
with open('oe.joblib', 'rb') as f2:
    oe = joblib.load(f2)

 
app=Flask(__name__)

@app.route('/')
def default():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['Age'])
        sex = str(request.form['Sex'])
        bmi = float(request.form['bmi'])
        smoker = str(request.form['Smoker'])
        region = str(request.form['region'])
        children = int(request.form['children'])
        region_2 = str(request.form['region_2'])


        features =  [age, sex, bmi, smoker, region, children, region_2]
        df = pd.DataFrame([features], columns=['age', 'sex', 'bmi', 'smoker', 'region', 'children', 'region_2'])
        df[['sex', 'smoker', 'region', 'region_2']] = oe.fit_transform(df[['sex', 'smoker', 'region', 'region_2']] )
        pred = model.predict(df)
        pred = round(float(pred[0]), 2)
        return render_template('index.html', pred_text=f"The predicted charges for medical insurance is : ${pred}")

    else:
        return render_template('index.html')

        
    
if __name__ =='__main__':
    app.run(debug=True)  