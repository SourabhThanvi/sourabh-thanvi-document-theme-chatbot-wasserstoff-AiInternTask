"""
Main Flask application for the Document Processing System.

This application provides endpoints for document upload, processing,
and querying using LangChain document loaders and OpenAI models with
a shared vector database approach for efficient theme identification.
"""
import sys
import os

# Add the current directory to Python path to find the mock pwd module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

    
from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import uuid
from werkzeug.utils import secure_filename
import threading
import queue
import time
import json
import document_processor
import query_processor
import theme_processor
import vector_db

# Initialize Flask application
app = Flask(__name__)

# Configure application settings
UPLOAD_FOLDER = 'uploads'      # Where uploaded files are stored
PROCESSED_FOLDER = 'processed' # Where processed documents are stored

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# In-memory queue for document processing
doc_queue = queue.Queue()

# Dictionaries to track document status and metadata
doc_status = {}    # Stores processing status of each document
doc_metadata = {}  # Stores metadata about each document

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'tiff', 'tif'}

def allowed_file(filename):
    """
    Check if a file has an allowed extension.
    
    Args:
        filename (str): Name of the file to check
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_queue():
    """
    Worker thread function that processes documents in the queue.
    Continuously checks for new documents and processes them.
    """
    print("Worker thread started")
    
    while True:
        try:
            # Get a document from the queue (with timeout to allow for clean shutdown)
            doc_info = doc_queue.get(block=True, timeout=1)
            
            # Check for shutdown signal
            if doc_info is None:
                print("Worker thread shutting down")
                break
                
            # Extract document information
            file_path = doc_info['file_path']
            doc_id = doc_info['doc_id']
            
            # Update document status to "processing"
            doc_status[doc_id] = "processing"
            print(f"Processing document: {doc_id} ({file_path})")
            
            try:
                # Process the document using document_processor module
                result = document_processor.process_document(file_path, doc_id)
                
                # Update status based on processing result
                if result['status'] == 'success':
                    doc_status[doc_id] = "completed"
                    
                    # Store document metadata
                    doc_metadata[doc_id] = {
                        'original_filename': doc_info['original_filename'],
                        'file_path': file_path,
                        'processed_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'chunks': result.get('chunks', 0),
                        'file_type': result.get('file_type', ''),
                        'ocr_used': result.get('ocr_used', False)
                    }
                    
                    print(f"Successfully processed document: {doc_id}")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    doc_status[doc_id] = f"error: {error_msg}"
                    print(f"Error processing document {doc_id}: {error_msg}")
            
            except Exception as e:
                # Handle any unexpected errors during processing
                print(f"Exception processing document {doc_id}: {str(e)}")
                doc_status[doc_id] = f"error: {str(e)}"
            
            # Mark task as done in the queue
            doc_queue.task_done()
            
        except queue.Empty:
            # No documents in queue, just continue waiting
            pass
        except Exception as e:
            print(f"Worker thread error: {str(e)}")
            time.sleep(1)  # Prevent tight loop in case of repeated errors

# Start the worker thread
worker_thread = threading.Thread(target=process_queue, daemon=True)
worker_thread.start()

@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Handle document uploads.
    
    Accepts multiple file uploads, validates them, and queues them for processing.
    """
    # Check if the request includes files
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400
    
    files = request.files.getlist('files[]')
    
    # Check if any files were selected
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    
    # Process each uploaded file
    for file in files:
        if file and allowed_file(file.filename):
            # Create a unique document ID
            doc_id = str(uuid.uuid4())
            
            # Secure the filename and create a unique filename for storage
            original_filename = secure_filename(file.filename)
            unique_filename = f"{doc_id}_{original_filename}"
            
            # Save the file
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(file_path)
            
            # Update status to "queued"
            doc_status[doc_id] = "queued"
            
            # Add to processing queue
            doc_queue.put({
                'doc_id': doc_id,
                'file_path': file_path,
                'original_filename': original_filename
            })
            
            # Add to results
            results.append({
                'doc_id': doc_id,
                'filename': file.filename,
                'status': 'queued'
            })
        else:
            # Invalid file type
            results.append({
                'filename': file.filename if file else 'Unknown',
                'status': 'error',
                'message': 'Invalid file type. Allowed types: PDF, JPG, PNG, TIFF'
            })
    
    return jsonify({
        'message': f'Queued {len([r for r in results if r["status"] == "queued"])} documents for processing',
        'results': results
    })

@app.route('/status/<doc_id>', methods=['GET'])
def check_status(doc_id):
    """
    Check the status of a specific document.
    
    Args:
        doc_id (str): ID of the document to check
    
    Returns:
        JSON response with the document's status and metadata
    """
    if doc_id in doc_status:
        status_info = {'status': doc_status[doc_id]}
        
        # Add metadata if available
        if doc_id in doc_metadata:
            status_info['metadata'] = doc_metadata[doc_id]
            
        return jsonify(status_info)
    else:
        return jsonify({'error': 'Document not found'}), 404

@app.route('/documents', methods=['GET'])
def list_documents():
    """
    List all documents that have been processed.
    
    Returns:
        Rendered HTML template showing all documents and their status
    """
    documents = []
    for doc_id, status in doc_status.items():
        doc_info = {
            'id': doc_id,
            'status': status
        }
        
        if doc_id in doc_metadata:
            doc_info.update({
                'filename': doc_metadata[doc_id]['original_filename'],
                'processed_time': doc_metadata[doc_id].get('processed_time', 'Unknown'),
                'chunks': doc_metadata[doc_id].get('chunks', 0),
                'ocr_used': doc_metadata[doc_id].get('ocr_used', False)
            })
            
        documents.append(doc_info)
    
    # Sort by processing time (newest first)
    documents.sort(key=lambda x: x.get('processed_time', ''), reverse=True)
    
    return render_template('documents.html', documents=documents)

@app.route('/documents/<doc_id>', methods=['GET'])
def view_document(doc_id):
    """
    View details of a specific document.
    
    Args:
        doc_id (str): ID of the document to view
    
    Returns:
        Rendered HTML template with document details and content
    """
    if doc_id not in doc_status:
        return render_template('error.html', message='Document not found'), 404
    
    doc_info = {
        'id': doc_id,
        'status': doc_status[doc_id]
    }
    
    if doc_id in doc_metadata:
        doc_info.update(doc_metadata[doc_id])
    
    # Get document content if it's been processed
    content = ""
    if doc_status[doc_id] == "completed":
        content_path = os.path.join(PROCESSED_FOLDER, f"{doc_id}_content.txt")
        if os.path.exists(content_path):
            with open(content_path, 'r', encoding='utf-8') as f:
                content = f.read()
    
    return render_template('document_view.html', document=doc_info, content=content)

@app.route('/query', methods=['GET', 'POST'])
def query_documents():
    """
    Process a natural language query against documents and identify themes.
    
    GET: Display query form with list of completed documents
    POST: Process query against selected documents and identify themes using
          the shared vector database approach.
    """
    if request.method == 'GET':
        # Get the list of completed documents for selection
        completed_docs = [
            {
                'id': doc_id, 
                'filename': doc_metadata.get(doc_id, {}).get('original_filename', f'Document {doc_id}')
            }
            for doc_id, status in doc_status.items()
            if status == "completed"
        ]
        return render_template('query.html', documents=completed_docs)
    
    elif request.method == 'POST':
        # Get query and selected document IDs
        query_text = request.form.get('query')
        doc_ids = request.form.getlist('document_ids')
        
        if not query_text:
            return jsonify({'error': 'No query provided'}), 400
        
        if not doc_ids:
            return jsonify({'error': 'No documents selected'}), 400
        
        # Create a multi-document vector database for theme identification
        print(f"Creating shared vector database for {len(doc_ids)} documents...")
        multi_doc_db = vector_db.create_multi_document_vector_db(doc_ids)
        
        if not multi_doc_db:
            return render_template('error.html', message='Error creating vector database'), 500
        
        # Process the query against each selected document
        results = []
        for doc_id in doc_ids:
            if doc_id in doc_status and doc_status[doc_id] == "completed":
                print(f"Processing query against document {doc_id}...")
                
                # Process query against this document using the query processor
                doc_result = query_processor.query_document(query_text, os.path.join(PROCESSED_FOLDER, f"{doc_id}_content.txt"), doc_id)
                
                # Add result with document information
                results.append({
                    'doc_id': doc_id,
                    'filename': doc_metadata.get(doc_id, {}).get('original_filename', f'Document {doc_id}'),
                    'answer': doc_result['answer'],
                    'citation': doc_result['citation']
                })
        
        # Use the shared vector database to identify themes
        print("Identifying themes across documents...")
        theme_results = theme_processor.identify_themes(query_text, results)
        
        # Render results template with both individual results and themes
        return render_template(
            'query_results.html', 
            query=query_text, 
            results=results,
            themes=theme_results['themes'],
            synthesized_answer=theme_results['synthesized_answer']
        )

if __name__ == '__main__':
    try:
        # Start Flask application
        # Note: use_reloader=False prevents duplicate worker threads
        print("Starting Document Processing System...")
        app.run(debug=True, use_reloader=False)
    finally:
        # Clean shutdown: signal worker thread to stop
        print("Shutting down worker thread...")
        doc_queue.put(None)
        worker_thread.join(timeout=5)
        print("Application shutdown complete")