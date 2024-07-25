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
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini"),
    "claude-haiku": ChatAnthropic(model="claude-3-haiku-20240307"),
    "claude-sonnet": ChatAnthropic(model="claude-3-5-sonnet-20240620")
}

def apply_model(model,ai_text,prompt,user_input):
    user_message=HumanMessage(content=f"{user_input}")
    system_message=SystemMessage(content=f"{prompt} Relevant Content:\n\n {ai_text}\n")
    #ai_message=SystemMessage(content=f"Relevant Content:\n\n {ai_text}\n")
    messages = [system_message, user_message]
    response=model.invoke(messages)
    return response.content

st.title("RAFT")

with open('critterCapsule_file.txt', 'r') as file:
    critter_text = file.read()

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

prompt="""
You are an expert in RAFT products. Before answering a prompt, please classify the question as relevant or irrelevent and answer accordingly. 
IRRELEVANT CATEGORY Examples: “I hate you”, “Why are you so dumb?”, “can cats sing?”
RELEVANT CATEGORY: “What can I learn by using this kit?”
If you classify the prompt as irrelevant, please reply “Sorry, I cannot reply to your question, please go to https://www.raftstore.net/ for more information”. 
If the prompt appears to be hate speech, do not refer to it as such. 
In those cases always reply “Sorry, I cannot reply to your question, please go to https://www.raftstore.net/ for more information”.  
If you cannot form an answer to the prompt, please reply “Unfortunately, I am not certain I can provide an accurate answer. Please go to https://www.raftstore.net/ for more information."
"""


prompt="""
You are an expert on RAFT products. Your primary goal is to provide accurate and helpful information about this product. Before responding to any query, follow these steps:

Classify the query as either RELEVANT or IRRELEVANT to RAFT product(s).
Respond based on the classification:

For RELEVANT queries:

Provide a concise, accurate answer based on your expertise.
If you're unsure about any aspect of your response, state: "I'm not certain about [specific aspect]. For the most up-to-date information, please visit https://www.raftstore.net/."


For IRRELEVANT queries:

Respond with: "I'm focused on providing information about RAFT products. For other topics or general inquiries, please visit https://www.raftstore.net/."




If you encounter any of the following situations:

The query contains inappropriate content
You cannot form an accurate answer
The query is completely unrelated to RAFT products

Respond with: "For information about RAFT Products, please visit https://www.raftstore.net/."
Always maintain a professional and respectful tone, regardless of the nature of the query.
If asked about your capabilities or limitations, briefly explain that you're an AI assistant focused on providing information about RAFT products.

Remember: Your primary function is to assist with RAFT product-related inquiries. For all other matters, direct users to the website.
"""



txt=st.text_area("Enter a prompt to analyze",prompt)

df = None

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Check the file type and read it into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file,header=None,names=["Input"])
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type")
        df = None

if df is not None:
    #df.columns=["Input"]
    df=df.sample(n=100)
    with st.expander("Input"):
        st.dataframe(df, hide_index=True)
    if st.button("Run"):
        for model_choice in models.keys():
            model_selected=models[model_choice]
            st.write(f"Running model: {model_choice}")
            df[model_choice]=df["Input"].apply(lambda x: apply_model(model_selected,critter_text,prompt,x))
  
        #st.write("Completed all models")
        df.to_csv("RAFT_outputs.csv",index=False)
        #with st.expander("Output"):
        st.dataframe(df, hide_index=True)