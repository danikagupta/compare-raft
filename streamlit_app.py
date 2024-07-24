import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

import pandas as pd

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

models={
    "gpt-4o": ChatOpenAI(model="gpt-4o"),
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o"),
    "claude-haiku": ChatAnthropic(model="claude-3-haiku-20240307"),
    "claude-sonnet": ChatAnthropic(model="claude-3-5-sonnet-20240620")
}

def apply_model(model,prompt,user_input):
    user_message=HumanMessage(content=f"{user_input}")
    system_message=SystemMessage(content=f"{prompt}")
    messages = [system_message, user_message]
    response=model.invoke(messages)
    return response.content

st.title("RAFT")
st.write("Hello World!!")

prompt="""
You are an expert in RAFT products. 
Before answering a prompt, please classify the question as relevant or irrelevent and answer accordingly. 
If you classify the prompt as irrelevent, 
please reply “Sorry, I cannot reply to your question, please go to https://www.raftstore.net/ for more information”. 
If you cannot form an answer to the prompt, 
please reply “Unfortunately, I am not certain I can provide an accurate answer. Please go to https://www.raftstore.net/ for more information ”. 
If none of these conditions apply, please answer the question in the same age level as the question - that is, assess the age level of the person asking the question and reply at the same level. 
If you cannot determine the age level of the questioner, reply in a way an elementary school student will understand.
"""

df=pd.read_csv("RAFT_inputs.csv")
df.columns=["Input"]
df=df.sample(n=4)
with st.expander("Input"):
    st.dataframe(df, hide_index=True)
if model_choice := st.radio("Choose a model",models.keys(),index=None):
    model_selected=models[model_choice]
    st.write(f"Selected model: {model_choice}")
    df[model_choice]=df["Input"].apply(lambda x: apply_model(model_selected,prompt,x))   
st.write("Done")
df.to_csv("RAFT_outputs.csv",index=False)
with st.expander("Output"):
    st.dataframe(df, hide_index=True)