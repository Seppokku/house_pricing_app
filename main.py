import streamlit as st
import pickle
import joblib
import pandas as pd
import sklearn
import numpy as np
from custom_transformers import NewFeatures
from catboost import CatBoostRegressor
import category_encoders

sklearn.set_config(transform_output='pandas')
st.title('House price prediction app')
st.subheader('Kaggle score')
st.image('image.png')

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')


cat_boost_sasha_pipeline = joblib.load('ml_pipeline.pkl')


data = st.file_uploader("Upload a CSV")

if data:
    data_to_predict = pd.read_csv(data)
    st.subheader('Your data before:')
    st.write(data_to_predict.head())

    button = st.button('Make prediction')
    if button:
        prediction = cat_boost_sasha_pipeline.predict(data_to_predict)
        pred_df = pd.DataFrame({'Id' : data_to_predict['Id'], 'SalePrice':np.round(np.exp(prediction), 2)})
        st.subheader('Your predictions:')
        st.write(pred_df.head())

        st.subheader('Download it to use on kaggle submitions')

        st.download_button(
        label="Download predictions as CSV",
        data=convert_df_to_csv(pred_df),
        file_name="predictions.csv",
        mime="text/csv")

