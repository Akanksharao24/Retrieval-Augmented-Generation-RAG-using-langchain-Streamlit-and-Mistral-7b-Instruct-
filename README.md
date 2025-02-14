# Retrieval-Augmented Generation (RAG) using LangChain, Streamlit, and Mistral-7B-Instruct

## Overview

This project implements a **lightweight, real-time Retrieval-Augmented Generation (RAG) system** that enables users to upload multiple **unrelated multi-page PDF documents** and extract relevant contextual information based on a query. The system efficiently handles **structured and unstructured PDFs** like invoices, reports, legal papers, and contracts.

## Features

✅ **Independent Document Handling**: Each PDF is treated as an independent knowledge source, ensuring no cross-document contamination.

✅ **Contextualized Responses**: Extracted answers indicate the section in the document where the response is found.

✅ **Efficient Processing**: Prioritizes important sections of large PDFs rather than blindly scanning all pages.

✅ **Fast Indexing & Caching**: Already uploaded PDFs are indexed for quick retrieval, avoiding redundant processing.

✅ **Summarization of Extracted Data**: Generates human-readable summaries instead of raw values.

✅ **Confidence Scores**: Provides a confidence score based on text clarity and contextual relevance.

---

## Tech Stack

- **LangChain**: For implementing Retrieval-Augmented Generation (RAG).
- **Streamlit**: For building an interactive web application.
- **Mistral-7B-Instruct**: A powerful open-source LLM for conversational responses.
- **FAISS**: For fast and efficient vector-based document retrieval.
- **HuggingFace Embeddings**: To generate text embeddings for document indexing.
- **PyPDFLoader**: For loading and processing PDF documents.
- **pytesseract & pdf2image**: To handle scanned PDFs using OCR (Optical Character Recognition).

---

## Installation & Setup

### Prerequisites

Ensure you have **Python 3.8+** installed. You also need to install dependencies using the following steps:

```bash
# Clone the repository
git clone https://github.com/your-repo/RAG-Langchain-Streamlit.git
cd RAG-Langchain-Streamlit

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

```

### Running the Application

After installation, run the Streamlit application using:

```bash
streamlit run app.py
```

This will launch the web app in your browser, where you can upload PDFs and start querying them.

---

## How It Works

1. **Upload PDFs**: Users can upload multiple unrelated PDFs.
2. **Text Extraction**:
   - Extracts text from structured PDFs using `PyPDFLoader`.
   - Uses OCR (`pytesseract`) for scanned PDFs.
3. **Text Processing**:
   - Splits text into manageable chunks using `RecursiveCharacterTextSplitter`.
   - Converts text into vector embeddings using `HuggingFace Embeddings`.
   - Stores embeddings in a **FAISS** vector store for fast retrieval.
4. **Conversational Querying**:
   - Uses `ConversationalRetrievalChain` from LangChain to retrieve relevant information.
   - The **Mistral-7B-Instruct** model generates responses based on retrieved text.
   - Summarizes responses and provides confidence scores.
5. **User Interaction**:
   - The chat interface, built with **Streamlit**, allows users to ask questions and receive contextual responses.

---

## Expected Output

- **High retrieval accuracy**, ensuring only relevant information is extracted.
- **Fast response times**, optimized by caching and efficient indexing.
- **Human-readable summarized answers** rather than raw data.
- **Confidence scores** to indicate result reliability.
  
---

### User Interface
Below are examples of different types of uploaded documents and queries:

#### Email Extraction
![Email Query Example](https://raw.githubusercontent.com/Akanksharao24/Retrieval-Augmented-Generation-RAG-using-langchain-Streamlit-and-Mistral-7b-Instruct-/main/Images/1.jpg)
![Email Query Example](https://raw.githubusercontent.com/Akanksharao24/Retrieval-Augmented-Generation-RAG-using-langchain-Streamlit-and-Mistral-7b-Instruct-/main/Images/2.jpg)

#### Invoice Processing
![Invoice Example](https://raw.githubusercontent.com/Akanksharao24/Retrieval-Augmented-Generation-RAG-using-langchain-Streamlit-and-Mistral-7b-Instruct-/main/Images/3.jpg)

#### Storybook Analysis
![Storybook Example](https://raw.githubusercontent.com/Akanksharao24/Retrieval-Augmented-Generation-RAG-using-langchain-Streamlit-and-Mistral-7b-Instruct-/main/Images/4.jpg)


---

## Team - Dead Strings

👩‍💻 **Sree Chakritha**\
👩‍💻 **Thrishita**\
👩‍💻 **Sriya**\
👩‍💻 **Akanksha**

---

## Future Enhancements

🔹 Implement **metadata tagging** for better search refinement. 🔹 Support **multimodal documents** (images, tables, etc.). 🔹 Deploy as a **web service** for broader accessibility.


---

Enjoy using the **PDF ChatBot with Mistral-7B-Instruct**! 🚀

