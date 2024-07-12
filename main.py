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
st.subheader('Our Kaggle results')

col_s, col_l = st.columns(2)

col_s.header('Sasha')
col_s.subheader(f'Score: 0.12355')
col_s.subheader(f'Place: 473')

col_l.header('Luba')
col_l.subheader(f'Score: 0.12508')
col_l.subheader(f'Place: 569')

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')


cat_boost_sasha_pipeline = joblib.load('ml_pipeline.pkl')
cat_boost_luba_pipeline = joblib.load('ml_pipeline_2.pkl')


data = st.file_uploader("Upload a CSV")

if data:
    data_to_predict = pd.read_csv(data)
    st.subheader('Your data before:')
    st.write(data_to_predict.head())

    button = st.button('Make prediction')
    if button:
        prediction_sasha = cat_boost_sasha_pipeline.predict(data_to_predict)
        prediction_luba = cat_boost_luba_pipeline.predict(data_to_predict)
        pred_df_sasha = pd.DataFrame({'Id' : data_to_predict['Id'], 'SalePrice':np.round(np.exp(prediction_sasha), 2)})
        pred_df_luba = pd.DataFrame({'Id' : data_to_predict['Id'], 'SalePrice':np.round(np.exp(prediction_luba), 2)})
        st.subheader('Your predictions:')
        col1, col2 = st.columns(2)

        col1.subheader("Sasha's predictions")
        col1.write(pred_df_sasha.head())

        col2.subheader("Luba's predictions")
        col2.write(pred_df_luba.head())


        st.subheader('Download it to use on kaggle submitions')

        col3, col4 = st.columns(2)


        col3.download_button(
        label="Download Sasha's predictions as CSV",
        data=convert_df_to_csv(pred_df_sasha),
        file_name="predictions_sasha.csv",
        mime="text/csv")

        col4.download_button(
        label="Download Luba's's predictions as CSV",
        data=convert_df_to_csv(pred_df_luba),
        file_name="predictions_luba.csv",
        mime="text/csv")




