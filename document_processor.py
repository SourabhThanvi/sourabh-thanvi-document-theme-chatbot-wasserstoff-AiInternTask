"""
Document processing module that handles document ingestion using LangChain.

This module provides functions to process various document types (PDF, images)
using appropriate LangChain document loaders, with intelligent OCR handling
for scanned documents.
"""

import os
import time
import json
import platform
# In document_processor.py
# Instead of:
# from langchain_community.document_loaders.pdf import PyPDFLoader

# Use:
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredImageLoader
from langchain_community.document_loaders import UnstructuredPDFLoader

# # Handle Windows-specific imports
# if platform.system() != 'Windows':
#     from langchain_community.document_loaders import (
#         PyPDFLoader, 
#         UnstructuredImageLoader, 
#         UnstructuredPDFLoader
#     )
# else:
#     # Windows-specific imports
#     from langchain_community.document_loaders.pdf import PyPDFLoader
#     from langchain_community.document_loaders.image import UnstructuredImageLoader
#     from langchain_community.document_loaders.pdf import UnstructuredPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_document(file_path, doc_id):
    """
    Process a document using the appropriate LangChain document loader based on type.
    
    Args:
        file_path (str): Path to the document file
        doc_id (str): Unique identifier for the document
        
    Returns:
        dict: Processing result with status and metadata
    """
    try:
        print(f"Starting processing for document: {file_path}")
        
        # Create directories for processed files
        processed_dir = 'processed'
        os.makedirs(processed_dir, exist_ok=True)
        
        # Get file extension to determine document type
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Initialize variable to track if OCR was used
        ocr_used = False
        
        # Choose appropriate loader based on file type
        if file_ext == '.pdf':
            # First try PyPDFLoader which is faster but only works for text-based PDFs
            try:
                print(f"Attempting to load PDF with PyPDFLoader: {file_path}")
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # Check if extracted content is meaningful
                total_text = "".join([doc.page_content for doc in documents])
                
                # If very little text was extracted, the PDF might be scanned/image-based
                if len(total_text.strip()) < 100:
                    print(f"Detected possible scanned PDF (minimal text content), switching to OCR: {file_path}")
                    loader = UnstructuredPDFLoader(file_path, mode="elements")
                    documents = loader.load()
                    ocr_used = True
                else:
                    print(f"Successfully loaded text-based PDF: {file_path}")
            except Exception as e:
                # If PyPDFLoader fails, try UnstructuredPDFLoader with OCR
                print(f"PyPDFLoader failed, using UnstructuredPDFLoader with OCR: {str(e)}")
                loader = UnstructuredPDFLoader(file_path, mode="elements")
                documents = loader.load()
                ocr_used = True
                
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            # Use UnstructuredImageLoader which applies OCR for images
            print(f"Processing image with OCR: {file_path}")
            loader = UnstructuredImageLoader(file_path)
            documents = loader.load()
            ocr_used = True
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Create a text splitter for chunking (for better retrieval later)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,         # Size of each chunk in characters
            chunk_overlap=200,       # Overlap between chunks to maintain context
            separators=["\n\n", "\n", ".", " ", ""]  # Hierarchy of separators
        )
        
        # Split documents into chunks
        if documents:
            split_docs = text_splitter.split_documents(documents)
            print(f"Document split into {len(split_docs)} chunks")
        else:
            # If no documents were loaded, create an empty list
            print(f"Warning: No content extracted from {file_path}")
            split_docs = []
        
        # Prepare metadata and content for saving
        pages_info = []
        full_text = ""
        
        for i, doc in enumerate(split_docs):
            # Ensure page information is in metadata
            if 'page' not in doc.metadata:
                doc.metadata['page'] = i + 1
                
            # Add document ID to metadata
            doc.metadata['doc_id'] = doc_id
                
            # Create page/chunk info
            page_info = {
                'chunk_id': i + 1,
                'page': doc.metadata.get('page', 'Unknown'),
                'source': file_path,
                'doc_id': doc_id
            }
            pages_info.append(page_info)
            
            # Add to full text with chunk markers for citation
            full_text += f"\n--- Document Chunk {i+1} ---\n"
            full_text += doc.page_content
            full_text += "\n"
        
        # Save the processed content as plain text
        content_path = os.path.join(processed_dir, f"{doc_id}_content.txt")
        with open(content_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        # Save chunk information as JSON for structured access
        chunks_path = os.path.join(processed_dir, f"{doc_id}_chunks.json")
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(pages_info, f, indent=2)
        
        # Save document metadata
        metadata_path = os.path.join(processed_dir, f"{doc_id}_metadata.json")
        metadata = {
            'doc_id': doc_id,
            'original_file': os.path.basename(file_path),
            'file_type': file_ext,
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'chunks_count': len(split_docs),
            'ocr_used': ocr_used
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Successfully processed document {doc_id} with {len(split_docs)} chunks")
        
        # Return success result with metadata
        return {
            'status': 'success',
            'doc_id': doc_id,
            'chunks': len(split_docs),
            'file_type': file_ext,
            'ocr_used': ocr_used
        }
    
    except Exception as e:
        # Log and return error information
        print(f"Error processing document {doc_id}: {str(e)}")
        return {
            'status': 'error',
            'doc_id': doc_id,
            'error': str(e)
        }