import streamlit as st
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings

# Simple tokenizer function to replace NLTK
def simple_tokenizer(text):
    return text.lower().split()

# Pinecone API Key (For demo purposes, it is hardcoded. Replace with a secure method in production)
api_key = "39f61a31-5175-4eab-a795-6958263612f9"

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

# Initialize BM25Encoder with the simple tokenizer
bm25_encoder = BM25Encoder(tokenizer=simple_tokenizer).default()

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
if st.button("Add Sample Texts"):
    texts = [
        "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",
    ]
    retriever.add_texts(texts)
    st.write("Sample texts added to the retriever.")

if __name__ == "__main__":
    st.write("Streamlit app is running...")
