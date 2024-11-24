import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai  as genai
import app

def model_llm(combined_data):
  spliter=RecursiveCharacterTextSplitter(separators=['\n','.',' '],chunk_size=1000,chunk_overlap=100)
  chunks=[spliter.split_documents([d])for d in combined_data]

  chunks_flatten=[chunk for sublist in chunks for chunk in sublist] 

  embed=SentenceTransformer("all-MiniLM-L6-v2")
  app.placeholder.text("Embedding 🔢🔢🔢▶️")
  vectors=[embed.encode(chunk.page_content)for chunk in chunks_flatten]
  
  app.placeholder.text("What are your Queries❓❓❓")
  messages = st.container(border=True)
  if prompt := st.chat_input("Say something"):
      messages.chat_message("user").write(prompt)
      app.placeholder.text("Embedding 🔢🔢🔢▶️")
      x=embed.encode(prompt)
      x_reshaped = x.reshape(1, -1)
      app.placeholder.text("Finding cosine-similarities from the data💢💢💢")
      co=cosine_similarity(x_reshaped,vectors)
      para=chunks_flatten[co.argmax()]


      generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
      }

      model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=f"find answer of this question- {prompt} only answer from this paragarph {para}",
      )
      
      
      ans=model.generate_content(prompt)

      
      messages.chat_message("assistant").write(f"Echo: {ans.text}")
      app.placeholder.text("What are your Queries❓❓❓")
  return
  