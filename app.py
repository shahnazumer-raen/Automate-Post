import streamlit as st
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Qdrant # used for creating the Qdrant Vector Object
from langchain_community.embeddings.openai import OpenAIEmbeddings # used for embeddings 
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client.http import models
from langchain.llms import OpenAI
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
import qdrant_client
import os


def get_vector_store():
    
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )
    
    return vector_store

#define a function to create a Vector store
def create_vector_store(name_input):
    global Q_client
    embeddings=OpenAIEmbeddings(openai_api_key=st.secrets['OPENAI_API_KEY'])

    try:
        vector_store = Qdrant(
            client=Q_client, 
            collection_name=name_input, 
            embeddings=embeddings,
        )
    except:
        st.write('Failed to create a vector store')
    return vector_store


def main():
    load_dotenv()
    
    st.set_page_config(page_title="Ask Me")
    st.header("Ask your remote database 💬")
    
    # create vector store
    vector_store = get_vector_store()
    
    # create chain 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # show user input
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        st.write(f"Question: {user_question}")
        answer = qa.run(user_question)
        st.write(f"Answer: {answer}")
    
        
if __name__ == '__main__':
    main()