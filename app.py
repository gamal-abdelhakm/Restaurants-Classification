import streamlit as st
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("zomato_cleaned.csv")
model = joblib.load('models/model.h5')
scaler = joblib.load('models/scaler.h5')
listed_in_city_Encoder = joblib.load('models/listed_in_city_Encoder.h5')
listed_in_type_Encoder = joblib.load('models/listed_in_type_Encoder.h5')
location_Encoder = joblib.load('models/location_Encoder.h5')
inp_data = []
result = ''

st.markdown("### Predict the success of your new restaurant in Bangalore ")
st.markdown('______________________________________')
col1, col2 = st.columns(2)

with col1:
    f4 = st.text_input('Enter an approximate price for two', '')
    f3 = st.multiselect('Select your restaurant location',df['location'].unique(),max_selections=1)
    f5 = st.multiselect('Select your restaurant listed_in(type)', df['listed_in(type)'].unique(), max_selections=1)
    f1 = 1 if st.checkbox("Will your restaurant support online ordering?", False) else 0
with col2:
    f6 = st.multiselect('Select your restaurant listed_in(city)', df['listed_in(city)'].unique(), max_selections=1)
    f7 = st.multiselect('Choose your rest_type', list(df.columns)[8:32])
    f8 = st.multiselect('Choose your cuisines', list(df.columns)[32:])
    f2 = 1 if st.checkbox("Will your restaurant support table booking?", False) else 0

with col1:
    if st.button(' Predict '):
        inp_data.append(f1)
        inp_data.append(f2)
        inp_data.append(int(location_Encoder.transform([f3[0]])[0]))
        inp_data.append(int(f4))
        inp_data.append(int(listed_in_type_Encoder.transform([f5[0]])[0]))
        inp_data.append(int(listed_in_city_Encoder.transform([f6[0]])[0]))
        inp_data.append(len(f8))
        inp_data.extend([1 if x in f7 else 0 for x in list(df.columns)[8:32]])
        inp_data.extend([1 if x in f8 else 0 for x in list(df.columns)[32:]])
        result = np.array(inp_data).reshape(1, 134)
        result = scaler.transform([inp_data])
        result = model.predict(result)[0]


with col2:
    if result == "Yes":
        result = '<p style="font-family:Verdana;color:Green; font-size: 20px;">It will succeed insha allah 😊</p>'
        st.markdown(result,unsafe_allow_html=True)
    elif result == "No":
        result = '<p style="font-family:Verdana; color:Red; font-size: 20px;">Sorry it will fail 🥺</p>'
        st.markdown(result,unsafe_allow_html=True)
