from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
# load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    age =float(request.form['age'])
    workclass =float(request.form['workclass'])
    fnlwgt =float(request.form['fnlwgt'])
    education =float(request.form['education'])
    educationalnum =float(request.form['educational-num'])
    maritalstatus =float(request.form['marital-status'])
    occupation =float(request.form['occupation'])
    relationship =float(request.form['relationship'])
    race =float(request.form['race'])
    gender =float(request.form['gender'])
    capitalgain =float(request.form['capital-gain'])
    capitalloss =float(request.form['capital-loss'])
    hoursperweek =float(request.form['hours-per-week'])
    nativecountry =float(request.form['native-country'])

    result = model.predict([[age, workclass, fnlwgt, education,	educationalnum, maritalstatus, occupation, relationship, race, gender, capitalgain, capitalloss, hoursperweek, nativecountry]])[0]
    return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)