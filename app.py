import streamlit as st
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Pinecone API Key (For demo purposes, it is hardcoded. Replace with a secure method in production)
api_key = os.getenv("API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)
index_name = "hybrid-search-langchain-pinecone"

# Create the Pinecone index if it does not exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Load the index
index = pc.Index(index_name)

# Initialize embeddings and BM25 encoder
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25_encoder = BM25Encoder().default()

# Create the retriever
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

# Streamlit UI
st.title("Hybrid Search with LangChain & Pinecone")

# Input query
query = st.text_input("Enter your search query:")

if st.button("Search"):
    if query:
        result = retriever.invoke(query)
        st.write("**Search Result:**", result)
    else:
        st.write("Please enter a query to search.")

# Optionally, add texts to the retriever
st.subheader("Add Sample Texts")

# Text input for adding new texts
sample_texts_input = st.text_area("Enter your sample texts, one per line:")

if st.button("Add Texts"):
    if sample_texts_input:
        texts = [line.strip() for line in sample_texts_input.split('\n') if line.strip()]
        retriever.add_texts(texts)
        st.write(f"{len(texts)} sample texts added to the retriever.")
    else:
        st.write("Please enter some texts to add.")

if __name__ == "__main__":
    st.write("Streamlit app is running...")
