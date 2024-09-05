import pandas as pd
import numpy as np
import pickle as pkl
import streamlit as st

# Load pre-trained model and scaler
with open('.pkl/scaler.pkl', 'rb') as f:
    mms = pkl.load(f)

with open('.pkl/regressor.pkl', 'rb') as f:
    model_xgb = pkl.load(f)

# Manual mappings for categorical variables
gender_mapping = {'f': 0, 'm': 1}
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

# Streamlit app setup
st.title('Engineering Graduate Salary Prediction')

Gender = st.selectbox('Gender', ['f', 'm'])
p10percentage = st.number_input('P10 Percentage', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
p12percentage = st.number_input('P12 Percentage', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
CollegeTier = st.slider('College Tier', min_value=1, max_value=2, value=1, step=1)
Degree = st.selectbox('Degree', ['B.Tech/B.E.', 'M.Tech./M.E.', 'MCA', 'M.Sc. (Tech.)'])
Specialization = st.selectbox('Specialization', ['Other', 'computer science & engineering', 'electronics & telecommunications', 'mechanical engineering', 'information technology', 'electronics and communication engineering', 'computer engineering', 'computer application', 'electrical engineering', 'electronics and electrical engineering', 'electronics & instrumentation eng'])
collegeGPA = st.number_input('College GPA', min_value=0.0, max_value=100.0, value=0.0, step=0.5)

English = st.number_input('English', min_value=0, max_value=650, value=0, step=1)
Logical = st.number_input('Logical', min_value=0, max_value=1000, value=0, step=1)
Quant = st.number_input('Quant', min_value=0, max_value=1000, value=0, step=1)

Domain = st.number_input('Domain', min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
ComputerProgramming = st.number_input('Computer Programming', min_value=-1.0, max_value=1000.0, value=0.0, step=1.0)
ElectronicsAndSemicon = st.number_input('Electronics and Semicon', min_value=-1.0, max_value=650.0, value=0.0, step=1.0)
ComputerScience = st.number_input('Computer Science', min_value=-1.0, max_value=1000.0, value=0.0, step=1.0)
MechanicalEngg = st.number_input('Mechanical Engg', min_value=-1.0, max_value=1000.0, value=0.0, step=1.0)
ElectricalEngg =  st.number_input('Electrical Engg', min_value=-1.0, max_value=1000.0, value=0.0, step=1.0)
TelecomEngg = st.number_input('Telecom Engg', min_value=-1.0, max_value=1000.0, value=0.0, step=1.0)
CivilEngg = st.number_input('Civil Engg', min_value=-1.0, max_value=1000.0, value=0.0, step=1.0)

conscientiousness = st.number_input('Conscientiousness', min_value=-4.0, max_value=2.1, value=0.0, step=2.5)
agreeableness = st.number_input('Agreeableness', min_value=-6.0, max_value=2.0, value=0.0, step=2.5)
extraversion = st.number_input('Extraversion', min_value=-5.0, max_value=2.5, value=0.0, step=1.0)
nueroticism =  st.number_input('Nueroticism', min_value=-3.0, max_value=4.0, value=0.0, step=1.0)
openess_to_experience = st.number_input('Openess to Experience', min_value=-8.0, max_value=2.0, value=0.0, step=1.0)

columns = ['Gender', '10percentage', '12percentage', 'CollegeTier', 'Degree',
       'Specialization', 'collegeGPA', 'English', 'Logical', 'Quant', 'Domain',
       'ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience',
       'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg',
       'conscientiousness', 'agreeableness', 'extraversion', 'nueroticism',
       'openess_to_experience']

# label mapping
gender_mapping = {'f': 0, 'm': 1}
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
    st.success(f'Predicted Salary: {prediction[0]}')

st.info('Copyright © Ardava Barus - All rights reserved')