import streamlit as st
import joblib
# import Relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Title
st.title("Diabetes Prediction Web App")

# Sidebar
st.header("User Input")
st.subheader("Enter your details")

# User Input Form
form = st.form('data_form')
HighBP = form.selectbox("High Blood Pressure", ("Yes", "No"), placeholder="Choose an option")
HighChol = form.selectbox("High Cholesterol", ["Yes", "No"])
CholCheck = form.selectbox("Have you check Cholesterol in the last 5 years", ["Yes", "No"])
BMI = form.number_input("Enter your BMI",min_value=10, max_value=60, value='min')
Smoker = form.selectbox("Have you smoked up to 100 packs of ciggarette in your entire life", ["Yes", "No"])
Stroke = form.selectbox("Ever told you had stroke?", ["Yes", "No"])
HeartDiseaseorAttack = form.selectbox("Coronary heart disease (CHD) or Myocardial infarction (MI)", ["Yes", "No"])
PhysActivity = form.selectbox("Physical activity in the last 30 days", ["Yes", "No"])
Fruits = form.selectbox("Consume Fruit 1 or more times per day", ["Yes", "No"])
Veggies = form.selectbox("Consume Vegetables 1 or more times per day", ["Yes", "No"])
HvyAlcoholConsump = form.selectbox("Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)", ["Yes", "No"])
AnyHealthcare = form.selectbox("Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc", ["Yes", "No"])
NoDocbcCost = form.selectbox("Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?", ["Yes", "No"])
GenHlth = form.selectbox("How would you describe your general health", ["Excellent", "Very good","Good","Fair","Poor"])
MentHlth = form.number_input("How many days in the last 30 days was your mental health not good?", min_value=0, max_value=30, value='min')
PhysHlth = form.number_input("How many days in the last 30 days was your physical health not good?", min_value=0, max_value=30, value='min')
DiffWalk = form.selectbox("Do you have serious difficulty walking or climbing stairs?", ["Yes", "No"])
Sex = form.selectbox("Select your gender", ["Male", "Female"])
Age = form.selectbox("Select Age Category", ["18-24", "24-30","30-36","36-42","42-48","48-54","54-58",
                                            "58-60","60-64","64-70","70-76","76-80","80+"])
Education = form.selectbox("Highest Level of Education",["Never attended school",
                                                       "Elementary",
                                                       "Some high school",
                                                       "High school graduate",
                                                       "Some college or technical school",
                                                       "College graduate"
                                                      ])
Income = form.selectbox("Income Scale",['less than $10,000','$10,000-$15,000','$15,000-$20,000',
                                      '$20,000-$25,000','$25,000-$35,000','$35,000-$55,000',
                                      '$55,000-$75,000','$75,000 or more'])
form.form_submit_button('Get Prediction')




# Load the trained model and scaler
model = joblib.load("C:/Users/USER/Downloads/Diabetes_Project/model.pkl")  # Replace with your model filename
scaler = joblib.load("C:/Users/USER/Downloads/Diabetes_Project/new_scaler.pkl")  # Replace with your scaler filename

# Data Frame for user input
user_data = pd.DataFrame({
    "HighBP": [HighBP],
    "HighChol": [HighChol],
    "CholCheck": [CholCheck],
    "BMI": [BMI],
    "Smoker": [Smoker],
    "Stroke": [Stroke],
    "BMI": [BMI],
    "HeartDiseaseorAttack": [HeartDiseaseorAttack],
    "PhysActivity" : [PhysActivity],
    'Fruits' : [Fruits], 
    'Veggies' : [Veggies],
       'HvyAlcoholConsump' : [HvyAlcoholConsump], 
       'AnyHealthcare' : [AnyHealthcare], 
       'NoDocbcCost' : [NoDocbcCost],
        'GenHlth' : [GenHlth],
       'MentHlth' : [MentHlth], 
       'PhysHlth' : [PhysHlth],
       'DiffWalk' : [DiffWalk],
       'Sex':[Sex], 
       'Age' :[Age], 
       'Education': [Education],
       'Income':[Income]
})

yes_no_mapping = {'Yes': 1, 'No': 0}

# Apply the mapping to the selected columns
user_data['HighBP'] = user_data['HighBP'].map(yes_no_mapping)
user_data['HighChol'] = user_data['HighChol'].map(yes_no_mapping)
user_data['CholCheck'] = user_data['CholCheck'].map(yes_no_mapping)
user_data['Smoker'] = user_data['Smoker'].map(yes_no_mapping)
user_data['Stroke'] = user_data['Stroke'].map(yes_no_mapping)
user_data['HeartDiseaseorAttack'] = user_data['HeartDiseaseorAttack'].map(yes_no_mapping)


user_data['PhysActivity'] = user_data['PhysActivity'].map(yes_no_mapping)
user_data['Fruits'] = user_data['Fruits'].map(yes_no_mapping)
user_data['Veggies'] = user_data['Veggies'].map(yes_no_mapping)
user_data['HvyAlcoholConsump'] = user_data['HvyAlcoholConsump'].map(yes_no_mapping)
user_data['AnyHealthcare'] = user_data['AnyHealthcare'].map(yes_no_mapping)
user_data['NoDocbcCost'] = user_data['NoDocbcCost'].map(yes_no_mapping)


gen_hlth_mapping = {'Excellent': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5}

# Apply the mapping to the 'GenHlth' column
user_data['GenHlth'] = user_data['GenHlth'].map(gen_hlth_mapping)

user_data['DiffWalk'] = user_data['DiffWalk'].map(yes_no_mapping)
sex_mapping = {'Female': 0, 'Male': 1}
user_data['Sex'] = user_data['Sex'].map(sex_mapping)

education_mapping = {
    'Never attended school': 1,
    'Elementary': 2,
    'Some high school': 3,
    'High school graduate': 4,
    'Some college or technical school': 5,
    'College graduate': 6
}

# Apply the mapping to the 'Education' column
user_data['Education'] = user_data['Education'].map(education_mapping)
# Define mappings for 'Income' values to their corresponding numeric values
income_mapping = {
    'less than $10,000': 1,
    '$10,000-$15,000': 2,
    '$15,000-$20,000': 3,
    '$20,000-$25,000': 4,
    '$25,000-$35,000': 5,
    '$35,000-$55,000': 6,
    '$55,000-$75,000': 7,
    '$75,000 or more': 8
}
# Apply the mapping to the 'Income' column
user_data['Income'] = user_data['Income'].map(income_mapping)
age_group_mapping = {
    "18-24":1, "24-30":2,"30-36":3,"36-42":4,"42-48":5,"48-54":6,"54-58":7,
                                            "58-60":8,"60-64":9,"64-70":10,"70-76":11,"76-80":12,"80+":13
}
user_data['Age'] = user_data['Age'].map(age_group_mapping)
st.write(user_data)

cat_data = ['HighBP', 'HighChol', 'CholCheck','Smoker', 'Stroke',
       'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'DiffWalk', 'Sex', 'Age', 'Education',
       'Income']
user_data[cat_data] = user_data[cat_data].astype(np.int64)
user_data[['BMI','MentHlth','PhysHlth']] = user_data[['BMI','MentHlth','PhysHlth']].astype(np.float64)
# Apply scaling to user input
user_data[['BMI','MentHlth','PhysHlth']] = scaler.transform(user_data[['BMI','MentHlth','PhysHlth']])

# Make predictions for user data
prediction = model.predict(user_data)

# Display Prediction
st.subheader("Diabetes Prediction:")
if st.sidebar.button("Predict"):
    if prediction[0] == 0:
        st.write("Based on the input data, you are not likely to have diabetes.")
    elif prediction[0] == 1:
        st.write("Based on the input data, you are prediabetic.")
    elif prediction[0] == 2:
        st.write("Based on the input data, you have diabetes")

# About Section
st.sidebar.subheader("About")
st.sidebar.write(
    "This web app is designed to predict the likelihood of a person having diabetes "
    "based on their health data. The model is pre-trained and uses a scaler for input scaling."
)

# Add Your Own Style (CSS)
st.markdown(
    """
    <style>
    .st-bq {
        background-color: #f3f3f3;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 3px 3px 8px grey;
    }
    </style>
    """,
    unsafe_allow_html=True
)
