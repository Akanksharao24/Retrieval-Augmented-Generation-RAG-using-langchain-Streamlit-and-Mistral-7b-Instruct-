import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
import platform
from tempfile import TemporaryDirectory
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def extract_text_from_pdf(pdf_path):
    text = ""
    with TemporaryDirectory() as tempdir:
        if platform.system() == "Windows":
            pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            path_to_poppler_exe = Path(r"C:\\Program Files\\poppler-24.08.0\\Library\\bin")
            pdf_pages = convert_from_path(pdf_path, 500, poppler_path=path_to_poppler_exe)
        else:
            pdf_pages = convert_from_path(pdf_path, 500)
        
        for page_enumeration, page in enumerate(pdf_pages, start=1):
            filename = f"{tempdir}/page_{page_enumeration:03}.jpg"
            page.save(filename, "JPEG")
            text += str(pytesseract.image_to_string(Image.open(filename)))
    return text.replace("-\n", "")

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    llm = LlamaCpp(
        streaming=True,
        model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.75,
        top_p=1, 
        verbose=True,
        n_ctx=4096
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    return chain

def main():
    initialize_session_state()
    st.title("PDF ChatBot using Mistral-7B-Instruct :books:")
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                extracted_text = extract_text_from_pdf(temp_file_path)
                text.append(extracted_text)
                os.remove(temp_file_path)
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        text_chunks = text_splitter.split_text(" ".join(text))

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            model_kwargs={'device': 'cpu'}
        )
        
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        chain = create_conversational_chain(vector_store)

        display_chat_history(chain)

if __name__ == "__main__":
    main()
