"""
vector_db.py - Shared vector database functionality for document processing.

This module handles the creation, storage, and retrieval of vector embeddings
for document chunks to support both querying and theme identification.
"""

import os
import json
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Initialize the embeddings model using HuggingFace
embeddings = HuggingFaceEmbeddings()

def get_document_chunks(content_path, doc_id, chunks_path=None):
    """
    Load document content and convert to a list of Document objects.
    
    Args:
        content_path (str): Path to the document content file
        doc_id (str): Document identifier
        chunks_path (str, optional): Path to the document chunks JSON file
        
    Returns:
        list: List of Document objects with content and metadata
    """
    try:
        # Load document content
        with open(content_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Load document chunks metadata for citation information
        chunks_metadata = {}
        if not chunks_path:
            chunks_path = os.path.join('processed', f"{doc_id}_chunks.json")
        
        if os.path.exists(chunks_path):
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_info = json.load(f)
                # Create lookup by chunk ID
                for chunk in chunks_info:
                    chunk_id = chunk.get('chunk_id')
                    if chunk_id:
                        chunks_metadata[chunk_id] = chunk
        
        # Split text into chunks based on the chunk markers
        chunks = re.split(r'\n---\s*Document Chunk \d+\s*---\n', content)
        # Remove any empty chunks (e.g., before the first marker)
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        # Create document chunks with metadata
        docs = []
        
        for i, text_chunk in enumerate(chunks):
            # Get the chunk ID from the original document
            chunk_id = i + 1
            
            # Get metadata for this chunk from the lookup table
            metadata = chunks_metadata.get(chunk_id, {})
            if not metadata:
                # If metadata not found, create default metadata
                metadata = {
                    'page': 'Unknown',
                    'doc_id': doc_id,
                    'chunk_id': chunk_id
                }
            
            # Format citation string
            citation = f"Page {metadata.get('page', 'Unknown')}, Chunk {chunk_id}"
            metadata['citation'] = citation
            
            # Create a document object with text and metadata
            docs.append(Document(
                page_content=text_chunk.strip(),
                metadata=metadata
            ))
        
        return docs
    
    except Exception as e:
        print(f"Error loading document chunks for {doc_id}: {str(e)}")
        return []

def create_document_vector_db(doc_id):
    """
    Create a vector database for a single document.
    
    Args:
        doc_id (str): Document identifier
        
    Returns:
        FAISS: Vector store with document chunks or None if error
    """
    try:
        content_path = os.path.join('processed', f"{doc_id}_content.txt")
        chunks_path = os.path.join('processed', f"{doc_id}_chunks.json")
        
        # Get document chunks
        docs = get_document_chunks(content_path, doc_id, chunks_path)
        
        if not docs:
            print(f"No chunks found for document {doc_id}")
            return None
        
        # Create vector store
        vector_db = FAISS.from_documents(docs, embeddings)
        
        return vector_db
    
    except Exception as e:
        print(f"Error creating vector DB for document {doc_id}: {str(e)}")
        return None

def create_multi_document_vector_db(doc_ids):
    """
    Create a vector database for multiple documents.
    
    Args:
        doc_ids (list): List of document identifiers
        
    Returns:
        FAISS: Vector store with all documents or None if error
    """
    try:
        all_docs = []
        
        for doc_id in doc_ids:
            content_path = os.path.join('processed', f"{doc_id}_content.txt")
            docs = get_document_chunks(content_path, doc_id)
            all_docs.extend(docs)
        
        if not all_docs:
            print("No chunks found for any document")
            return None
        
        # Create vector store
        vector_db = FAISS.from_documents(all_docs, embeddings)
        
        return vector_db
    
    except Exception as e:
        print(f"Error creating multi-document vector DB: {str(e)}")
        return None

def query_vector_db(vector_db, query_text, k=3):
    """
    Query the vector database for relevant documents.
    
    Args:
        vector_db (FAISS): Vector store to query
        query_text (str): Query text
        k (int): Number of results to return
        
    Returns:
        list: List of relevant Document objects
    """
    if not vector_db:
        return []
    
    try:
        # Search for relevant documents
        relevant_docs = vector_db.similarity_search(query_text, k=k)
        return relevant_docs
    
    except Exception as e:
        print(f"Error querying vector DB: {str(e)}")
        return []