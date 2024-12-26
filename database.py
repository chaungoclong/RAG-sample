import os

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_vectorstore() -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Define the path where you want to save the FAISS index
    faiss_db_path = "./data"  # Choose a name for your index directory

    # Check if the index already exists
    if os.path.exists(faiss_db_path):
        # Load the existing index
        vectorstore = FAISS.load_local(faiss_db_path, embeddings, allow_dangerous_deserialization=True)
        print("Loaded existing FAISS index from disk.")
    else:
        loader = PDFPlumberLoader("./luat_dat_dai.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        # Create and save a new index
        vectorstore = FAISS.from_documents(all_splits, embeddings)  # use all_splits here
        vectorstore.save_local(faiss_db_path)
        print("Created and saved new FAISS index to disk.")

    return vectorstore
