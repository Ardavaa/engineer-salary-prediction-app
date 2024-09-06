import pandas as pd
import numpy as np
import pickle as pkl
import streamlit as st

# Load pre-trained model and scaler
with open('.pkl/scaler.pkl', 'rb') as f:
    mms = pkl.load(f)

with open('.pkl/regressor.pkl', 'rb') as f:
    model_xgb = pkl.load(f)

# Streamlit app setup
st.title('Engineering Graduate Salary Prediction')

Gender = st.selectbox('Gender', ['Female', 'Male'])
p10percentage = st.number_input('What is your overall marks obtained in grade 10 examinations? (P10 Percentage)', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
p12percentage = st.number_input('What is your overall marks obtained in grade 12 examinations? (P12 Percentage)', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
CollegeTier = st.slider('What is yourCollege Tier', min_value=1, max_value=2, value=1, step=1)
Degree = st.selectbox('What degree you obtained?', ['B.Tech/B.E.', 'M.Tech./M.E.', 'MCA', 'M.Sc. (Tech.)'])
Specialization = st.selectbox('What is your Specialization', ['Other', 'computer science & engineering', 'electronics & telecommunications', 'mechanical engineering', 'information technology', 'electronics and communication engineering', 'computer engineering', 'computer application', 'electrical engineering', 'electronics and electrical engineering', 'electronics & instrumentation eng'])
collegeGPA = st.number_input('What is your College GPA', min_value=0.0, max_value=100.0, value=0.0, step=0.5)

English = st.number_input('What is your English scores in AMCAT?', min_value=0, max_value=650, value=0, step=1)
Logical = st.number_input('What is your Logical scores in AMCAT?', min_value=0, max_value=1000, value=0, step=1)
Quant = st.number_input('What is your Quant scores in AMCAT?', min_value=0, max_value=1000, value=0, step=1)

Domain = st.number_input('What is your Domain scores in AMCAT?', min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
ComputerProgramming = st.number_input("What is your Computer Programming scores in AMCAT? (-1, if it's not available)", min_value=-1.0, max_value=1000.0, value=0.0, step=1.0)
ElectronicsAndSemicon = st.number_input("What is your Electronics and Semicon scores in AMCAT? (-1, if it's not available)", min_value=-1.0, max_value=650.0, value=0.0, step=1.0)
ComputerScience = st.number_input("What is your Computer Science scores in AMCAT? (-1, if it's not available)", min_value=-1.0, max_value=1000.0, value=0.0, step=1.0)
MechanicalEngg = st.number_input("What is your Mechanical Engineering scores in AMCAT? (-1, if it's not available)", min_value=-1.0, max_value=1000.0, value=0.0, step=1.0)
ElectricalEngg =  st.number_input("What is your Electrical Engineering scores in AMCAT? (-1, if it's not available)", min_value=-1.0, max_value=1000.0, value=0.0, step=1.0)
TelecomEngg = st.number_input("What is your Telecommucation Engineering scores in AMCAT? (-1, if it's not available)", min_value=-1.0, max_value=1000.0, value=0.0, step=1.0)
CivilEngg = st.number_input("What is your Civil Engineering scores in AMCAT? (-1, if it's not available)", min_value=-1.0, max_value=1000.0, value=0.0, step=1.0)

conscientiousness = st.number_input("What is your Conscientiousness scores in AMCAT's personality test?", min_value=-4.0, max_value=2.1, value=0.0, step=2.5)
agreeableness = st.number_input("What is your Agreeableness scores in AMCAT's personality test?", min_value=-6.0, max_value=2.0, value=0.0, step=2.5)
extraversion = st.number_input("What is your Extraversion scores in AMCAT's personality test?", min_value=-5.0, max_value=2.5, value=0.0, step=1.0)
nueroticism =  st.number_input("What is your Nueroticism scores in AMCAT's personality test?", min_value=-3.0, max_value=4.0, value=0.0, step=1.0)
openess_to_experience = st.number_input("What is your Openess to Experience scores in AMCAT's personality test?", min_value=-8.0, max_value=2.0, value=0.0, step=1.0)

columns = ['Gender', '10percentage', '12percentage', 'CollegeTier', 'Degree',
       'Specialization', 'collegeGPA', 'English', 'Logical', 'Quant', 'Domain',
       'ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience',
       'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg',
       'conscientiousness', 'agreeableness', 'extraversion', 'nueroticism',
       'openess_to_experience']

# label mapping
gender_mapping = {'Female': 0, 'Male': 1}
degree_mapping = {'B.Tech/B.E.': 0, 'M.Tech./M.E.': 1, 'MCA': 2, 'M.Sc. (Tech.)': 3}
specialization_mapping = {
    'Other': 0,
    'computer science & engineering': 1,
    'electronics & telecommunications': 2,
    'mechanical engineering': 3,
    'information technology': 4,
  'electronics and communication engineering': 5,
  'computer engineering': 6,
  'computer application': 7,
  'electrical engineering': 8,
  'electronics and electrical engineering': 9,
  'electronics & instrumentation eng': 10
}

Gender_mapped = gender_mapping[Gender]
Degree_mapped = degree_mapping[Degree]
Specialization_mapped = specialization_mapping[Specialization]


# input datdaframe
input_data = pd.DataFrame([[
    Gender_mapped,
    p10percentage,
    p12percentage,
    CollegeTier,
    Degree_mapped,
    Specialization_mapped,
    collegeGPA,
    English,
    Logical,
    Quant,
    Domain,
    ComputerProgramming,
    ElectronicsAndSemicon,
    ComputerScience,
    MechanicalEngg,
    ElectricalEngg,
    TelecomEngg,
    CivilEngg,
    conscientiousness,
    agreeableness,
    extraversion,
    nueroticism,
    openess_to_experience
]], columns=columns)

input_data_scaled = mms.transform(input_data)

# make prediction
if st.button('Predict'):
    prediction = model_xgb.predict(input_data_scaled)
    st.success(f'Predicted Salary: ₹{int(round(prediction[0]))}')

st.info('Copyright © Ardava Barus - All rights reserved')