# Bedrock Multi-Model RAG App - Compare responses from multiple Amazon Bedrock models

This Streamlit application enables users to perform document-based question answering using multiple Amazon Bedrock models simultaneously.
The app leverages RAG (Retrieval Augmented Generation) to provide accurate answers based on uploaded text documents while comparing response times and quality across different models.

The application provides a flexible and user-friendly interface for document analysis and question answering, with key features including:

- Support for multiple text file uploads and processing
- FAISS vector store creation for efficient document retrieval
- Simultaneous querying of multiple Amazon Bedrock models
- Response time measurement and comparison
- Configurable parameters for fine-tuning model behavior and document processing

## Repository Structure

```
.
├── README.md                   # Project documentation
├── requirements.txt           # Core Python dependencies
├── streamlit_app/            # Main application directory
│   ├── app.py               # Primary Streamlit application code
│   ├── requirements.txt     # Streamlit-specific dependencies
│   └── uploaded_txts/       # Directory for uploaded text documents
└── uploaded_txts/           # Backup directory for text documents
```

## Usage Instructions

### Prerequisites

- Python 3.7+
- AWS account with access to Amazon Bedrock
- AWS credentials configured locally
- Streamlit 1.26.0 or higher

Required Python packages:

```
streamlit>=1.26.0
boto3>=1.28.0
loguru>=0.7.0
pandas>=2.0.0
langchain>=0.1.0
langchain_community>=0.0.10
faiss-cpu>=1.7.4
```

### Installation

1. Clone the repository:

```bash
git clone git@github.com:Nyx-Data-AI/PoC-VectorSearchModelComparer-GpoSalinas.git
cd git@github.com:Nyx-Data-AI/PoC-VectorSearchModelComparer-GpoSalinas.git
```

2. Create and activate a virtual environment:

```bash
# MacOS/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
pip install -r streamlit_app/requirements.txt
```

### Quick Start

1. Configure AWS credentials:

```bash
aws configure
```

2. Launch the Streamlit app:

```bash
cd streamlit_app
streamlit run app.py
```

3. Access the application at `http://localhost:8501`

### More Detailed Examples

1. Uploading Documents:

```python
# Select text files through the Streamlit interface
# Supported files will be processed and added to the FAISS vector store
```

2. Configuring Models:

```python
# Select from available Bedrock models in the sidebar:
# - Claude-3 Haiku
# - Claude-3.5 Haiku
```

3. Asking Questions:

```python
# Enter your question in the main interface
# The app will query all selected models simultaneously
# Results will show response times and answers from each model
```

### Troubleshooting

1. AWS Credentials Issues

- Error: "No credentials found"
  - Verify AWS credentials are properly configured
  - Check `~/.aws/credentials` file
  - Run `aws configure` to set up credentials

2. Model Access Issues

- Error: "AccessDeniedException"
  - Verify AWS account has access to Bedrock models
  - Check IAM permissions for Bedrock service
  - Ensure selected region supports desired models

3. Document Processing Issues

- Error: "Failed to process document"
  - Verify text file encoding (UTF-8 required)
  - Check file permissions
  - Ensure file size is within limits

## Data Flow

The application processes documents and queries through a RAG pipeline, transforming raw text into searchable vectors for context-aware responses.

```ascii
[Text Files] -> [Document Loader] -> [Text Splitter] -> [FAISS Vector Store]
                                                              |
[User Query] -> [Query Processing] -> [Context Retrieval] -> [LLM] -> [Response]
```

Component Interactions:

1. Document Loader processes uploaded text files
2. Text Splitter chunks documents into manageable segments
3. FAISS Vector Store indexes document chunks for efficient retrieval
4. Query Processor formats user questions
5. Context Retrieval system finds relevant document segments
6. Multiple Bedrock models process queries with retrieved context
7. Response Aggregator collects and displays model outputs
