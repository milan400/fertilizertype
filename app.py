#Basic
import numpy as np
import pandas as pd
import os

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# Load Model
import pickle

import streamlit as st 

dirname = os.path.dirname(__file__)

filename = os.path.join(dirname, 'crop classifier.pkl')
clf = pickle.load(open(filename, 'rb'))

filename_type = os.path.join(dirname, 'fertilizer type.pkl')
clf_type = pickle.load(open(filename_type, 'rb'))

filename_amount = os.path.join(dirname, 'regression.pkl')
clf_amount = pickle.load(open(filename_amount, 'rb'))



dataset = pd.read_csv(os.path.join(dirname,'dataset.csv'))

standardScaler = StandardScaler()
standardScaler_type = StandardScaler()
standardScaler_amount = StandardScaler()


label_crop = LabelEncoder()
label_soil = LabelEncoder()
label_type = LabelEncoder()

dataset['Crop Type'] = label_crop.fit_transform(dataset['Crop Type'])
dataset['Soil Type'] = label_soil.fit_transform(dataset['Soil Type'])
dataset['Fertilizer Name'] = label_type.fit_transform(dataset['Fertilizer Name'])


dataset_type = dataset.drop(columns=['Fertilizer Name'])
dataset_type_scaled = standardScaler_type.fit_transform(dataset_type)


dataset_amount = dataset.drop(columns=['Nitrogen', 'Potassium', 'Phosphorous'])
dataset_amount_scaled = standardScaler_amount.fit_transform(dataset_amount)

dataset = dataset.drop(columns=['Nitrogen', 'Potassium', 'Phosphorous', 'Fertilizer Name','Crop Type'])
dataset_scaled = standardScaler.fit_transform(dataset)

def predict_cropname(X):
    
    """Let's predict crop name 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Temparature
        in: query
        type: number
        required: true
      - name: Humidity
        in: query
        type: number
        required: true
      - name: Moisture
        in: query
        type: number
        required: true
      - name: Soil Type
        in: query
        type: number
        required: true     
      - name: Nitrogen
        in: query
        type: number
        required: true
      - name: Potassium
        in: query
        type: number
        required: true
      - name: Phosphorous
        in: query
        type: number
        required: true                                     
    responses:
        200:
            description: The output values
        
    """
   
    prediction=clf.predict(X)

    return prediction, label_crop.inverse_transform(prediction)[0] 



def main():
    st.title("Eco Fertilizer")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Eco Fertilizer ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    Temparature = st.number_input("Temparature",step=1.,format="%.0f") 

    Humidity = st.number_input("Humidity",step=1.,format="%.0f") 

    Moisture = st.number_input("Moisture",step=1.,format="%.0f") 

    Soil_Type = st.radio("Soil Type", ('Sandy', 'Loamy', 'Black', 'Red', 'Clayey'))

    if(Soil_Type=='Black'):
      Soil_Type= 0
    elif(Soil_Type=='Clayey'):
      Soil_Type= 1
    elif(Soil_Type=='Loamy'):
      Soil_Type= 2
    elif(Soil_Type=='Red'):
      Soil_Type= 3
    elif(Soil_Type=='Sandy'):
      Soil_Type= 4
     
    Nitrogen = st.number_input("Nitrogen",step=1.,format="%.0f")

    Potassium = st.number_input("Potassium",step=1.,format="%.0f")

    Phosphorous = st.number_input("Phosphorous",step=1.,format="%.0f")


    inpu = [[Temparature, Humidity, Moisture, Soil_Type]]
    test_dataset = pd.DataFrame(inpu, columns=['Temparature', 'Humidity', 'Moisture', 'Soil Type'])
    test_dataset = standardScaler.transform(test_dataset)
    test_dataset = np.array(test_dataset)

    result=""
    result_fertilizer_type=""
    prediction_type=""
    prediction_amount=0.0

    if st.button("Predict"):
        result_type, result=predict_cropname(test_dataset)
        
        if(len(result_type)!=0):
          # Prediction for Fertilizer Type
          inpu_type = [[Temparature, Humidity, Moisture, Soil_Type, result_type[0], Nitrogen, Potassium, Phosphorous]]

          test_dataset_type = pd.DataFrame(inpu_type, columns=['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'])
          test_dataset_type = np.array(test_dataset_type)

          prediction_type= clf_type.predict(test_dataset_type)
          result_fertilizer_type =  label_type.inverse_transform(prediction_type)[0]

          # Prediction for Fertilizer Value
          inpu_value = [[Temparature, Humidity, Moisture, Soil_Type, result_type[0], prediction_type[0]]]

          test_dataset_amount = pd.DataFrame(inpu_value, columns=['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Fertilizer Name'])
          test_dataset_amount = standardScaler_amount.transform(test_dataset_amount)
          test_dataset_amount = np.array(test_dataset_amount)

          prediction_amount = clf_amount.predict(test_dataset_amount)[0]



    st.success("Crop Name: " + result)
    st.success("Fertilizer Type: " + result_fertilizer_type)
    st.success("Fertilizer Amount: %d" % int(prediction_amount))
    if st.button("About"):
        st.text("Eco Fertilizer")
        st.text("Master Thesis Project")

if __name__=='__main__':
    main()