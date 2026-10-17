import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

st.title("Titanic Passenger Survival Prediction")
st.write("Enter passenger details to predict survival.")

pclass_encoder = LabelEncoder()
pclass_encoder.fit(["First", "Second", "Third"])

gender_encoder = LabelEncoder()
gender_encoder.fit(["Male", "Female"])

sibling_encoder = LabelEncoder()
sibling_encoder.fit(["Zero", "One", "Two", "Three"])

embarked_encoder = LabelEncoder()
embarked_encoder.fit(["Southampton", "Cherbourg", "Queenstown"])

file = open("svc_trained_model.pkl", "rb")
model = pickle.load(file)
file.close()

pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"])
gender = st.selectbox("Gender", ["Male", "Female"])
sibling = st.selectbox("Number of Siblings", ["Zero", "One", "Two", "Three"])
embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

if st.button("Predict Survival"):

    pclass = pclass_encoder.transform([pclass])[0]
    gender = gender_encoder.transform([gender])[0]
    sibling = sibling_encoder.transform([sibling])[0]
    embarked = embarked_encoder.transform([embarked])[0]

    data = pd.DataFrame({
        "PClass": [pclass],
        "Gender": [gender],
        "Sibling": [sibling],
        "Embarked": [embarked]
    })

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Passenger Survived!")
    else:
        st.error("Passenger Did Not Survive!")
