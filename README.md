# Document Research & Theme Identification Chatbot

An AI-powered document processing system that can analyze large sets of documents, identify common themes, and provide detailed, cited responses to user queries.

## Demo Overview & Performance


https://github.com/user-attachments/assets/20bddc37-fcfe-4c4b-a747-6c49a9850440


This demonstration showcases the complete functionality of the Document Research & Theme Identification Chatbot, including document upload, processing, query execution, and theme synthesis capabilities.

**Processing Time:** The demonstration shows the complete workflow taking approximately 5 minutes to execute. This processing time is primarily due to:

- Large document file sizes requiring extensive text extraction and processing
- Vector store creation and indexing of multiple documents for semantic search
- Embedding generation and database storage optimization
- Initial model loading and system initialization

The one-time setup ensures optimal performance for subsequent queries and enables fast, accurate theme identification across the entire document corpus.

The system demonstrates production-ready capabilities suitable for research, legal document analysis, and enterprise knowledge management applications.

## 🚀 Features

- **Multi-Document Processing**: Upload and process 75+ documents in PDF and image formats
- **OCR Support**: Extract text from scanned documents using OCR technology
- **Semantic Search**: Query documents using natural language with vector-based similarity search
- **Theme Identification**: Automatically identify common themes across multiple documents
- **Citation Management**: Precise citations with page and paragraph references
- **Web Interface**: User-friendly web interface for document upload and querying
- **Real-time Processing**: Background document processing with status tracking

## 🏗️ Architecture

### Core Components

1. **Document Processor** (`document_processor.py`): Handles PDF and image processing with OCR
2. **Vector Database** (`vector_db.py`): Creates and manages vector embeddings for semantic search
3. **Query Processor** (`query_processor.py`): Processes natural language queries against documents
4. **Theme Processor** (`theme_processor.py`): Identifies themes across multiple documents
5. **Flask Application** (`app.py`): Main web application with REST API endpoints

### Technology Stack

- **Backend**: Python, Flask
- **AI/ML**: LangChain, OpenAI GPT, HuggingFace Embeddings
- **Vector Database**: FAISS
- **Document Processing**: PyPDF2, PIL
- **Frontend**: HTML, CSS, JavaScript

## 📋 Requirements

### Python Dependencies

```
flask
langchain
langchain-openai
langchain-community
langchain-huggingface
openai
faiss-cpu
numpy
python-dotenv
werkzeug
uuid
threading
queue
time
json
os
re
```

### System Requirements

- Python 3.8+
- OpenAI API Key
- Sufficient storage for document processing
- RAM: 4GB+ recommended for large document sets

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/document-theme-chatbot.git
   cd document-theme-chatbot
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🚀 Usage

### Starting the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Using the Web Interface

1. **Upload Documents**: 
   - Navigate to the home page
   - Select multiple PDF or image files
   - Click upload to queue documents for processing

2. **Monitor Processing**:
   - Visit `/documents` to see processing status
   - Documents show as "queued", "processing", "completed", or "error"

3. **Query Documents**:
   - Go to `/query` page
   - Select completed documents
   - Enter your natural language query
   - View individual results and identified themes

### API Endpoints

- `POST /upload` - Upload documents
- `GET /documents` - List all documents
- `GET /status/<doc_id>` - Check document status
- `POST /query` - Query documents (accepts JSON or form data)
- `GET /health` - Health check

## 📁 Project Structure

```
document-theme-chatbot/
├── app.py                 # Main Flask application
├── document_processor.py  # Document processing logic
├── query_processor.py     # Query processing and LLM integration
├── theme_processor.py     # Theme identification logic
├── vector_db.py          # Vector database management
├── templates/           # HTML templates
│   ├── index.html      # Upload page
│   ├── documents.html  # Document list
│   ├── query.html      # Query interface
│   └── query_results.html # Results display
├── uploads/            # Uploaded files (gitignored)
├── processed/          # Processed documents (gitignored)
├── vector_databases/   # Vector DB storage (gitignored)
├── requirements.txt   # Python dependencies
├── .env              # Environment variables (gitignored)
├── .gitignore        # Git ignore rules
└── README.md         # This file
```

## 🔧 Configuration

### Document Processing Settings

- **Chunk Size**: 500 characters (configurable in `vector_db.py`)
- **Chunk Overlap**: 50 characters
- **Supported Formats**: PDF, JPG, JPEG, PNG, TIFF
- **Max File Size**: 50MB (configurable in `app.py`)

### AI Model Configuration

- **LLM Model**: GPT-4o-mini-2024-07-18 (configurable)
- **Embeddings**: HuggingFace embeddings
- **Temperature**: 0 (for consistent results)
- **Vector Search**: FAISS similarity search

## 📊 Performance Considerations

- **Document Processing**: CPU-intensive, runs in background thread
- **Vector Database**: Memory usage scales with document count
- **Query Processing**: Network latency dependent on OpenAI API
- **Theme Identification**: Processing time increases with document count

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Errors**:
   - Verify API key in `.env` file
   - Check API quota and billing

2. **Memory Errors**:
   - Reduce batch size for large documents
   - Increase system RAM

3. **Vector Database Errors**:
   - Ensure FAISS is properly installed
   - Check disk space for vector storage

### Debug Mode

Enable detailed logging by running with debug mode:
```bash
FLASK_DEBUG=1 python app.py
```

## 🧪 Testing

### Manual Testing

1. Upload a few test documents
2. Wait for processing completion
3. Try various queries to test functionality


## 🚀 Deployment

### Local Development
- Run directly with `python app.py`
- Use `debug=True` for development

### Environment Variables for Production

```
OPENAI_API_KEY=your_api_key
```

