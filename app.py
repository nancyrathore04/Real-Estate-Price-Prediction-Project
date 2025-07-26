from flask import Flask, render_template, request
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load model and columns
model = pickle.load(open('real estate price prediction project/model/banglore_home_prices_model.pickle', 'rb'))
with open('real estate price prediction project/model/columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']
    locations = data_columns[3:]  # Assuming area, bhk, bath are first

@app.route('/')
def index():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bhk = int(request.form['bhk'])
    location = request.form['location']

    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = area
    x[1] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    prediction = model.predict([x])[0]
    price = round(prediction, 2)

    return render_template('index.html', locations=locations, result=price)

if __name__ == '__main__':
    app.run(debug=True)

