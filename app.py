import pickle
from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
## importing ridge regressor model and standard scaler pickle
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))
ridge_model=pickle.load(open('models/ridge.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():       # note: what url_for given in home.html in form action should be same as the function name 
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature')) # note: here this 'Temperature' etc are should match with the name given homepage form's name in input line
        RH=float(request.form.get('RH'))                   # and the oder in which these features are written this should be macthed with order in model traing dataset  
        Ws=float(request.form.get('Ws'))                  # if order changed model will not predict correct
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)#   here we are storing our predicted value in result variable

        return render_template('home.html',result=result[0])  # here our result will be in form of list and that will contain only one value so written as [0]
    else:
        return render_template('home.html')





if __name__=="__main__":
    app.run(host="0.0.0.0")  # flask application by default runs on 5000 port no but if want to change we can change and can give any another port no by writing port=port_no
