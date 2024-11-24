import streamlit as st
from langchain.document_loaders import PyPDFLoader,UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai  as genai
from dotenv import load_dotenv
import os
import helper


load_dotenv()

genai.configure(api_key=os.getenv('api_key'))


with st.sidebar:
  st.subheader('AI pdf/url Q/A chatbot')
  file=st.file_uploader('Uplod your PDF Here',type='pdf')
  data=st.text_input('Upload your URL Here')
  
placeholder=st.empty()
if not file and not data:
  placeholder.header("UPLOAD Something to Start!!!")
  
if file:
  placeholder.text("Loading PDF data ⏱️⏱️⏱️")
  with open('temp.pdf','wb') as f:
    f.write(file.read())
  loader=PyPDFLoader("temp.pdf")
  pdf_data=loader.load() 
  

if data:
  placeholder.text("Loading URL data ⏱️⏱️⏱️")
  loader=UnstructuredURLLoader(urls=[data])
  url_data=loader.load()
  

if file and len(data)>0:
    placeholder.text("Both PDF and URL data loadeding ⏱️⏱️⏱️ ")
    combined_data = pdf_data + url_data
    helper.model_llm(combined_data)  
elif file and len(data)==0:
    helper.model_llm(pdf_data)  
elif file is None and len(data)>0:
    helper.model_llm(url_data) 

