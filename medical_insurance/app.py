from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

app=Flask(__name__)

#load the training model and column names
model=joblib.load('rfmodel.pkl')
train_columns=joblib.load('train_columns.pkl')

def preprocess_user_input(age,sex,bmi,children,smoker,alcoholic,disease,region):
        sex = 1 if sex.lower() == 'male' else 0
        smoker = 1 if smoker.lower() == 'yes' else 0
        alcoholic = 1 if alcoholic.lower() == 'yes' else 0
        
        disease_mapping={'none':0,'cancer':1,'diabetes':2,'heart disease':3,'hypertension':4,'obesity':5}
        disease = disease_mapping[disease.lower()]

        region_mapping={'northwest':0,'northeast':1,'southeast':2,'southwest':3}
        region = region_mapping[region.lower()]  
        
        user_input= [age,sex,bmi,children,smoker,alcoholic,disease,region]
        return np.array(user_input).reshape(1,-1) 
        

@app.route('/')
def home():
        return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
        age=int(request.form['age'])
        sex=request.form['sex']
        bmi=float(request.form['bmi'])
        children=int(request.form['children'])
        smoker=request.form['smoker']
        alcoholic=request.form['alcoholic']
        disease=request.form['disease']
        region=request.form['region']
        
        user_input=preprocess_user_input(age,sex,bmi,children,smoker,alcoholic,disease,region)
        prediction=model.predict(user_input)[0]
        

    # Make predictions
     

        #print("prediction:",prediction)
        return render_template('result.html',prediction= prediction )

if __name__=='__main__':
        app.run(debug=True)
        
        
