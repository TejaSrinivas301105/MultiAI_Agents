"""
Vector Store Module — Handles PDF processing, embedding, and FAISS vector store management.
"""
import os
import hashlib
import time
from datetime import datetime
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter

VECTOR_DIR = os.path.join(os.path.dirname(__file__), "vector_store_data")
os.makedirs(VECTOR_DIR, exist_ok=True)


def get_embedding_model(api_key: str, retries: int = 3, backoff_factor: int = 2):
    """Create embedding model with retry logic."""
    for attempt in range(retries):
        try:
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
            )
            _ = embedding_model.embed_query("connectivity probe")
            return embedding_model
        except Exception as e:
            if attempt < retries - 1:
                delay = backoff_factor ** attempt
                time.sleep(delay)
            else:
                raise Exception(f"Failed to initialize embedding model after {retries} attempts: {e}")


def get_pdf_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def process_pdf(file_path: str, original_filename: str, embedding_model) -> FAISS:
    """Process a single PDF and return a FAISS vector store."""
    pdf_hash = get_pdf_hash(file_path)
    vector_path = os.path.join(VECTOR_DIR, pdf_hash)

    if os.path.exists(vector_path):
        try:
            db = FAISS.load_local(
                vector_path,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True,
            )
            return db
        except Exception:
            pass  # fall through to recompute

    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No content extracted from PDF: {original_filename}")
        
        for doc in documents:
            doc.metadata.update({
                "source": original_filename,
                "hash": pdf_hash,
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        
        if not docs:
            raise ValueError(f"No text chunks created from PDF: {original_filename}")
        
        db = FAISS.from_documents(docs, embedding_model)
        db.save_local(vector_path)
        return db
    except Exception as e:
        raise Exception(f"Failed to process PDF '{original_filename}': {str(e)}")


def build_retriever(file_paths: List[dict], embedding_model):
    """
    Build a merged FAISS retriever from multiple PDFs.
    file_paths: list of {"path": str, "name": str}
    """
    if not file_paths:
        raise ValueError("No files provided to build retriever")
    
    all_dbs = []
    errors = []
    
    for fp in file_paths:
        try:
            db = process_pdf(fp["path"], fp["name"], embedding_model)
            all_dbs.append(db)
        except Exception as e:
            errors.append(f"{fp['name']}: {str(e)}")
    
    if not all_dbs:
        error_msg = "Failed to process any PDFs. Errors: " + "; ".join(errors)
        raise Exception(error_msg)
    
    if errors:
        print(f"⚠️ Warning: Some PDFs failed to process: {'; '.join(errors)}")

    merged = all_dbs[0]
    for extra_db in all_dbs[1:]:
        merged.merge_from(extra_db)

    return merged.as_retriever(search_kwargs={"k": 5})
