import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd

app = Flask(__name__) #starting point of the application where it will run

regmodel = pickle.load(open('regmodel.pkl', 'rb')) #import model
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/') #local host of the url
def home():
    return render_template('home.html') #html page

#create api to use postman and any other tools that allows to send request to the app and we can get output
@app.route('/predict_api', methods=['POST'])
def predict_api():
    #whenever I hit predict_api, i need to make sure the input I give is in json format, which
    #is captured by the data key and stored in the data variable
    data = request.json['data']
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST']) #we provide input and we submit the form, take data here, predict and get result
def predict():
    data = [float(x) for x in request.form.values()] #convert each value from form into float then put into a list
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output= regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text=f"The house price prediction is {output}")
    #replace prediction_text in placegolder in home.html with the value


if __name__ == "__main__":
    app.run(debug=True)
