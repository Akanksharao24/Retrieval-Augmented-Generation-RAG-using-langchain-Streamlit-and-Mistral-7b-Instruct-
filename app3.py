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

# Initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = {}
    if 'generated' not in st.session_state:
        st.session_state['generated'] = {}
    if 'past' not in st.session_state:
        st.session_state['past'] = {}

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain, doc_name):
    reply_container = st.container()
    container = st.container()
    
    with container:
        with st.form(key=f'my_form_{doc_name}', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder=f"Ask about {doc_name}", key=f'input_{doc_name}')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'][doc_name])
            
            st.session_state['past'][doc_name].append(user_input)
            st.session_state['generated'][doc_name].append(output)
    
    if st.session_state['generated'][doc_name]:
        with reply_container:
            for i in range(len(st.session_state['generated'][doc_name])):
                message(st.session_state['past'][doc_name][i], is_user=True, key=f"{doc_name}_{i}_user", avatar_style="thumbs")
                message(st.session_state['generated'][doc_name][i], key=f"{doc_name}_{i}", avatar_style="fun-emoji")

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
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return chain

def main():
    initialize_session_state()
    st.title("PDF ChatBot using Mistral-7B-Instruct :books:")
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
                text = loader.load()
                os.remove(temp_file_path)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
                text_chunks = text_splitter.split_documents(text)

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
                vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
                chain = create_conversational_chain(vector_store)
                
                st.session_state['history'][file.name] = []
                st.session_state['generated'][file.name] = []
                st.session_state['past'][file.name] = []
                
                st.subheader(f"Chat with {file.name}")
                display_chat_history(chain, file.name)

if __name__ == "__main__":
    main()
