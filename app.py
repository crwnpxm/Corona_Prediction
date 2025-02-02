import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import pandas as pd

# importing Model
model = load_model("covid_prediction.keras")

def pickel_loader(pickler_name, input_data):
    with open(pickler_name+'.pkl', 'rb') as file:
        pickle_load = pickle.load(file)
        input_data[pickler_name] = pickle_load.transform([input_data[pickler_name]])
        return input_data

st.title('covid_Prediction')
binary_select=(True,False)
Cough_symptoms=st.checkbox('Do you have symptoms of cough?',binary_select)
Fever=st.checkbox('Do you have symptoms of Fever?',binary_select)
Sore_throat=st.checkbox('Do you have symptoms of Sore throat?',binary_select)
Shortness_of_breath=st.checkbox('Do you have symptoms of Shortness of breath?',binary_select)
Headache=st.checkbox('Do you have symptoms of Headache?',binary_select)
Age_60_above=st.radio('Is your age is Age 60 above?',('Yes','No'))
Sex=st.radio('Choose your gender?',('male','female'))
Known_contact=st.radio('Is there any covid patient on your circle',('others','Contact with confirmed'))

analysis= st.button('check the covid prediction', type='primary', use_container_width=True)

if analysis:
    input={
        'Cough_symptoms' : Cough_symptoms,
        'Fever' : Fever,
        'Sore_throat' : Sore_throat,
        'Shortness_of_breath' : Shortness_of_breath,
        'Headache' : Headache,
        'Age_60_above' : Age_60_above,
        'Sex' : Sex,
        'Known_contact' :Known_contact,
    }


    for key, value in input.items():
        input = pickel_loader(key, input)
    input_df = pd.DataFrame(input)

    with open('scaller.pkl', 'rb') as file:
        scaller_file = pickle.load(file)
        input_df = scaller_file.transform(input_df)
        predict = model.predict(input_df)
        if predict > .5:
            st.info(f'You are COVID Positive. Prediction: {int(predict[0][0] * 100)} %')
        else:
            st.info(f'You are not COVID Positive. Prediction: {int(predict[0][0] * 100)} %')


