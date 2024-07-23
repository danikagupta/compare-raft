from openai import OpenAI
import os
import streamlit as st


client=OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
st.write("Hello World!!")
