import numpy as np
import pickle
import streamlit as st

with open('Model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Scaler.pkl', 'rb') as s:
    scaler = pickle.load(s)

location_mapping = {
    "Poranki": 8,
    "Kankipadu": 5,
    "Benz Circle": 0,
    "Gannavaram": 2,
    "Rajarajeswari Peta": 9,
    "Gunadala": 4,
    "Gollapudi": 3,
    "Enikepadu": 1,
    "Vidhyadharpuram": 10,
    "Penamaluru": 7,
    "Payakapuram": 6
}

status_mapping = {
    "Resale": 2,
    "Under Construction": 3,
    "Ready to move": 1,
    "New": 0
}

direction_mapping = {
    "Not Mentioned": 0,
    "East": 1,
    "West": 3,
    "NorthEast": 2
}

property_type_mapping = {
    "Apartment": 0,
    "Independent Floor": 1,
    "Independent House": 2,
    "Residential Plot": 3
}

def predict(bed, bath, loc, size, status, face, Type):

    selected_location_numeric = location_mapping[loc]
    selected_status_numeric = status_mapping[status]
    selected_direction_numeric = direction_mapping[face]
    selected_property_type_numeric = property_type_mapping[Type]

    input_data = np.array([[bed, bath, selected_location_numeric, size, selected_status_numeric, 
               selected_direction_numeric, selected_property_type_numeric]])

    input_df = scaler.transform(input_data)

    prediction = model.predict(input_df)[0]

    return prediction

if __name__ == '__main__':
    st.header('House Price Prediction')

    bed = st.slider('No of Bedrooms', max_value=10, min_value=0, value=2)
    bath = st.slider('No of Bathooms', max_value=10, min_value=0, value=2)
    loc = st.selectbox('Select a Location', list(location_mapping.keys()))
    size = st.number_input('Enter Sq Feet', max_value=10000, min_value=100, value=1000, step=500)
    status = st.selectbox('Select a Status', list(status_mapping.keys()))
    face = st.selectbox('Select a Direction', list(direction_mapping.keys()))
    Type = st.selectbox('Select a Property Type', list(property_type_mapping.keys()))

    result = predict(bed, bath, loc, size, status, face, Type)

    st.write(f'The predicted value is : {result} Lakhs')
