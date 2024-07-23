from openai import OpenAI
from langchain import LangChain
import os
import streamlit as st

import pandas as pd

models={
    "gpt-4o": LangChain.from_pretrained('gpt-4o'),
    "gpt-4o-mini": LangChain.from_pretrained('gpt-4o-mini'),
    "claude-haiku": LangChain.from_pretrained('claude-haiku'),
    "claude-sonnet":LangChain.from_pretrained('claude-sonnet')
}


client=OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
st.write("Hello World!!")

df=pd.read_csv("RAFT_inputs.csv")
df.columns=["Prompt"]
with st.expander("Input"):
    st.dataframe(df, hide_index=True)
if model_choice := st.radio("Choose a model",models.keys(),index=None):
    model_selected=models[model_choice]
    st.write(f"Selected model: {model_choice}")
    df[model_choice]=df["Prompt"].apply(lambda x: model_selected(x))   
st.write("Done")
with st.expander("Output"):
    st.dataframe(df, hide_index=True)