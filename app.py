from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Train the model once when app starts
df = pd.read_csv('insurance.csv')
le_sex = LabelEncoder(); 
le_sex.fit(df['sex'])
le_smoker = LabelEncoder(); 
le_smoker.fit(df['smoker'])
le_region = LabelEncoder(); 
le_region.fit(df['region'])
df['sex'] = le_sex.transform(df['sex'])
df['smoker'] = le_smoker.transform(df['smoker'])
df['region'] = le_region.transform(df['region'])
X = df[['age','bmi','children','sex','smoker','region']]
y = df['expenses']
model = GradientBoostingRegressor(random_state=42).fit(X, y)

HTML_FORM = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Insurance Cost Predictor</title>
</head>
<body style="
    margin: 0;
    padding: 20px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #f8fafc;
    color: #334155;
">

<div style="
    max-width: 400px;
    margin: 0 auto;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    padding: 32px;
">
    
    <h1 style="
        text-align: center;
        color: #1e293b;
        font-size: 24px;
        font-weight: 600;
        margin: 0 0 24px 0;
    ">Insurance Cost Predictor</h1>

    <form method="post" style="display: flex; flex-direction: column; gap: 16px;">
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div>
                <label style="
                    display: block;
                    font-weight: 500;
                    margin-bottom: 6px;
                    color: #475569;
                ">Age</label>
                <input name="age" type="number" value="30" min="18" max="100" style="
                    width: 100%%;
                    padding: 10px 12px;
                    border: 1px solid #d1d5db;
                    border-radius: 6px;
                    font-size: 14px;
                    box-sizing: border-box;
                    transition: border-color 0.2s;
                " onfocus="this.style.borderColor='#3b82f6'" onblur="this.style.borderColor='#d1d5db'">
            </div>
            
            <div>
                <label style="
                    display: block;
                    font-weight: 500;
                    margin-bottom: 6px;
                    color: #475569;
                ">BMI</label>
                <input name="bmi" type="number" step="0.1" value="25" style="
                    width: 100%%;
                    padding: 10px 12px;
                    border: 1px solid #d1d5db;
                    border-radius: 6px;
                    font-size: 14px;
                    box-sizing: border-box;
                    transition: border-color 0.2s;
                " onfocus="this.style.borderColor='#3b82f6'" onblur="this.style.borderColor='#d1d5db'">
            </div>
        </div>

        <div>
            <label style="
                display: block;
                font-weight: 500;
                margin-bottom: 6px;
                color: #475569;
            ">Children</label>
            <input name="children" type="number" min="0" max="5" value="0" style="
                width: 100%%;
                padding: 10px 12px;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                font-size: 14px;
                box-sizing: border-box;
                transition: border-color 0.2s;
            " onfocus="this.style.borderColor='#3b82f6'" onblur="this.style.borderColor='#d1d5db'">
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div>
                <label style="
                    display: block;
                    font-weight: 500;
                    margin-bottom: 6px;
                    color: #475569;
                ">Gender</label>
                <select name="sex" style="
                    width: 100%%;
                    padding: 10px 12px;
                    border: 1px solid #d1d5db;
                    border-radius: 6px;
                    font-size: 14px;
                    box-sizing: border-box;
                    background: white;
                    transition: border-color 0.2s;
                " onfocus="this.style.borderColor='#3b82f6'" onblur="this.style.borderColor='#d1d5db'">
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                </select>
            </div>

            <div>
                <label style="
                    display: block;
                    font-weight: 500;
                    margin-bottom: 6px;
                    color: #475569;
                ">Smoker</label>
                <select name="smoker" style="
                    width: 100%%;
                    padding: 10px 12px;
                    border: 1px solid #d1d5db;
                    border-radius: 6px;
                    font-size: 14px;
                    box-sizing: border-box;
                    background: white;
                    transition: border-color 0.2s;
                " onfocus="this.style.borderColor='#3b82f6'" onblur="this.style.borderColor='#d1d5db'">
                    <option value="no">No</option>
                    <option value="yes">Yes</option>
                </select>
            </div>
        </div>

        <div>
            <label style="
                display: block;
                font-weight: 500;
                margin-bottom: 6px;
                color: #475569;
            ">Region</label>
            <select name="region" style="
                width: 100%%;
                padding: 10px 12px;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                font-size: 14px;
                box-sizing: border-box;
                background: white;
                transition: border-color 0.2s;
            " onfocus="this.style.borderColor='#3b82f6'" onblur="this.style.borderColor='#d1d5db'">
                <option value="southwest">Southwest</option>
                <option value="southeast">Southeast</option>
                <option value="northwest">Northwest</option>
                <option value="northeast">Northeast</option>
            </select>
        </div>

        <input type="submit" value="Predict Cost" style="
            background: #3b82f6;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 8px;
        " onmouseover="this.style.background='#2563eb'" onmouseout="this.style.background='#3b82f6'">
    </form>

    %s

</div>

</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = ""
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])
            sex = le_sex.transform([request.form['sex']])[0]
            smoker = le_smoker.transform([request.form['smoker']])[0]
            region = le_region.transform([request.form['region']])[0]
            features = np.array([[age, bmi, children, sex, smoker, region]])
            pred = model.predict(features)[0]
            result = f"""
            <div style="
                background: #f0f9ff;
                border: 1px solid #bae6fd;
                border-radius: 8px;
                padding: 16px;
                margin-top: 20px;
                text-align: center;
            ">
                <h3 style="margin: 0 0 8px 0; color: #1e40af;">Estimated Annual Cost</h3>
                <p style="font-size: 24px; font-weight: 600; margin: 0; color: #059669;">${pred:,.2f}</p>
            </div>
            """
        except Exception as e:
            result = f"""
            <div style="
                background: #fef2f2;
                border: 1px solid #fecaca;
                border-radius: 8px;
                padding: 16px;
                margin-top: 20px;
                color: #dc2626;
                text-align: center;
            ">
                <strong>Error:</strong> {e}
            </div>
            """
    return HTML_FORM % result

if __name__ == '__main__':
    app.run(debug=True)
