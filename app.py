import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
model=tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('one_hot_encoder_geography.pkl','rb') as file:
    ohe=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
st.title('Customer Churn Predicition')
geography=st.selectbox('Geography',ohe.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
est_salary=st.number_input("estimated salary")
tenure=st.slider("Tenure",0,10)
prod=st.slider("number of products",1,4)
has_cr_card=st.selectbox("has credit card",[0,1])
is_active=st.selectbox('Is active member',[0,1])
input_data=pd.DataFrame({'CreditScore':[credit_score],'Gender':[label_encoder_gender.transform([gender])[0]],'Age':[age],'Tenure':[tenure],'Balance':[balance],'NumOfProducts':[prod],'HasCrCard':[has_cr_card],'IsActiveMember':[is_active],'EstimatedSalary':[est_salary]})

geo_encoded=ohe.transform([[geography]])
geo_df=pd.DataFrame(geo_encoded.toarray(),columns=ohe.get_feature_names_out(['Geography']))
input_data=pd.concat([input_data.reset_index(drop=True),geo_df],axis=1)
input_data_scaled=scaler.transform(input_data)
pred=model.predict(input_data_scaled)
if pred>0.5:
    st.write("the employee is likely to churn")
else:
    st.write('the employee is not likely to churn')



