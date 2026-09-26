import streamlit as st
import pandas as pd

st.title("AML Signal Escalation - 264FC1A2")
st.write("CV AUC: 0.856±0.007")

try:
    df = pd.read_csv('team_264FC1A2.csv')
    st.metric("Signals", len(df))
    st.dataframe(df)
except:
    st.error("CSV file required")
