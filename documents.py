# # import os
# # import logging
# # import streamlit as st
# # from langchain_community.document_loaders import PyPDFLoader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_ollama import OllamaEmbeddings
# # from langchain_chroma import Chroma
# # from langchain_community.llms.ollama import Ollama
# # from langchain.chains.retrieval_qa.base import RetrievalQA
# # from langchain.prompts import PromptTemplate
# # from langsmith import traceable
# # from dotenv import load_dotenv
# # from tqdm import tqdm

# # load_dotenv(dotenv_path="./.env", verbose=True, override=True)

# # # Configure logging
# # logging.basicConfig(level=logging.INFO,
# #                     format='%(asctime)s - %(levelname)s - %(message)s')

# # @traceable(run_type="llm", metadata={"ls_provider": "ollama", "model": "mistral"})
# # def create_qa_agent(pdf_path, model_name="mistral"):
# #     persist_directory = "./data/chroma_db"

# #     if os.path.exists(persist_directory):
# #         logging.info("Loading existing Chroma store...")
# #         vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model=model_name))
# #     else:
# #         logging.info("Creating new Chroma store...")
# #         loader = PyPDFLoader(pdf_path)
# #         pages = loader.load()
# #         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300, length_function=len)
# #         splits = text_splitter.split_documents(pages)
# #         embeddings = OllamaEmbeddings(model=model_name)
# #         vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# #         for chunk in tqdm(splits, desc="Processing chunks"):
# #             vectorstore.add_documents([chunk], embedding=embeddings)

# #     llm = Ollama(model=model_name)

# #     prompt_template = """
# #     You are a helpful AI assistant that answers questions based on the provided PDF document.
# #     Use only the context provided to answer the question. If you don't know the answer, say so.
    
# #     Context: {context}
    
# #     Question: {question}
    
# #     Answer: """

# #     PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# #     qa_chain = RetrievalQA.from_chain_type(
# #         llm=llm,
# #         chain_type="stuff",
# #         retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
# #         return_source_documents=True,
# #         chain_type_kwargs={"prompt": PROMPT}
# #     )

# #     return qa_chain

# # @traceable(run_type="chain")
# # def ask_question(qa_chain, question):
# #     try:
# #         response = qa_chain({"query": question})
# #         return {
# #             "answer": response["result"],
# #             "sources": [doc.page_content for doc in response["source_documents"]]
# #         }
# #     except Exception as e:
# #         logging.error(f"An error occurred: {str(e)}")
# #         return {"error": str(e), "answer": None, "sources": None}

# # st.title("Chat with PDF using Niki AI")

# # uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# # if uploaded_file is not None:
# #     pdf_path = f"./temp_uploaded.pdf"
# #     with open(pdf_path, "wb") as f:
# #         f.write(uploaded_file.getbuffer())
    
# #     st.success("PDF uploaded successfully. Processing...")
# #     qa_agent = create_qa_agent(pdf_path)
# #     st.session_state["qa_agent"] = qa_agent

# # if "qa_agent" in st.session_state:
# #     user_input = st.text_area("Enter your question:")
# #     if st.button("Ask"):
# #         if user_input.strip():
# #             result = ask_question(st.session_state["qa_agent"], user_input)
# #             if result.get("error"):
# #                 st.error(result["error"])
# #             else:
# #                 st.subheader("Niki AI Response:")
# #                 st.write(result["answer"])
# #         else:
# #             st.warning("Please enter a question before asking.")

# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################

# # import os
# # import logging
# # import streamlit as st
# # from langchain_community.document_loaders import PyPDFLoader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_ollama import OllamaEmbeddings
# # from langchain_chroma import Chroma
# # from langchain_community.llms.ollama import Ollama
# # from langchain.chains.retrieval_qa.base import RetrievalQA
# # from langchain.prompts import PromptTemplate
# # from langsmith import traceable
# # from dotenv import load_dotenv
# # from tqdm import tqdm

# # # Configure logging
# # logging.basicConfig(level=logging.INFO,
# #                     format='%(asctime)s - %(levelname)s - %(message)s')

# # PERSIST_DIRECTORY = "./data/chroma_db"
# # PDF_STORAGE_PATH = "./temp_uploaded.pdf"

# # def pdf_exists():
# #     """Check if a PDF has ever been uploaded."""
# #     return os.path.exists(PERSIST_DIRECTORY) or os.path.exists(PDF_STORAGE_PATH)

# # @traceable(run_type="llm", metadata={"ls_provider": "ollama", "model": "mistral"})
# # def create_qa_agent(pdf_path, model_name="mistral"):
# #     if os.path.exists(PERSIST_DIRECTORY):
# #         logging.info("Loading existing Chroma store...")
# #         vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=OllamaEmbeddings(model=model_name))
# #     else:
# #         logging.info("Creating new Chroma store...")
# #         loader = PyPDFLoader(pdf_path)
# #         pages = loader.load()
# #         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300, length_function=len)
# #         splits = text_splitter.split_documents(pages)
# #         embeddings = OllamaEmbeddings(model=model_name)
# #         vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

# #         for chunk in tqdm(splits, desc="Processing chunks"):
# #             vectorstore.add_documents([chunk], embedding=embeddings)

# #     llm = Ollama(model=model_name)

# #     prompt_template = """
# #     You are a helpful AI assistant that answers questions based on the provided PDF document.
# #     Use only the context provided to answer the question. If you don't know the answer, say so.
    
# #     Context: {context}
    
# #     Question: {question}
    
# #     Answer: """

# #     PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# #     qa_chain = RetrievalQA.from_chain_type(
# #         llm=llm,
# #         chain_type="stuff",
# #         retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
# #         return_source_documents=True,
# #         chain_type_kwargs={"prompt": PROMPT}
# #     )

# #     return qa_chain

# # @traceable(run_type="chain")
# # def ask_question(qa_chain, question):
# #     try:
# #         response = qa_chain({"query": question})
# #         return {
# #             "answer": response["result"],
# #             "sources": [doc.page_content for doc in response["source_documents"]]
# #         }
# #     except Exception as e:
# #         logging.error(f"An error occurred: {str(e)}")
# #         return {"error": str(e), "answer": None, "sources": None}

# # st.title("Chat with PDF using Niki AI")

# # # User input field
# # user_input = st.text_area("Enter your question:")

# # # Buttons for Upload and Ask
# # col1, col2 = st.columns([4, 1])

# # with col1:
# #     if st.button("Ask"):
# #         if not user_input.strip():
# #             st.warning("Please enter a question before asking.")
# #         elif not pdf_exists():
# #             st.warning("Please upload a PDF to proceed.")
# #         else:
# #             if "qa_agent" not in st.session_state:
# #                 st.session_state["qa_agent"] = create_qa_agent(PDF_STORAGE_PATH)
# #             result = ask_question(st.session_state["qa_agent"], user_input)
# #             if result.get("error"):
# #                 st.error(result["error"])
# #             else:
# #                 st.subheader("Niki AI Response:")
# #                 st.write(result["answer"])

# # with col2:
# #     if st.button("Upload PDF"):
# #         uploaded_file = st.file_uploader("Choose a PDF to upload", type=["pdf"], key="pdf_uploader", label_visibility="collapsed")
# #         if uploaded_file:
# #             with open(PDF_STORAGE_PATH, "wb") as f:
# #                 f.write(uploaded_file.getbuffer())
# #             st.success("PDF uploaded successfully. Processing...")
# #             st.session_state["qa_agent"] = create_qa_agent(PDF_STORAGE_PATH)








# #             ##############################
# #             #                           
# #             # 
# #             #   
# #             # 
# #             # 
# #             # 
# #             #       Fix the ./.env shit
# #             #
# #             #
# #             #
# #             #
# #             #
# #             #
# #             #
# #             #############################

# #fixed version. from nv

import os
import logging
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langsmith import traceable
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

PERSIST_DIRECTORY = "./data/chroma_db"
PDF_STORAGE_PATH = "./data/uploads"

os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

def pdf_exists():
    """Check if any PDF exists in the storage path."""
    return bool(os.listdir(PDF_STORAGE_PATH))

def save_uploaded_file(uploaded_file):
    """Save the uploaded PDF to the designated directory."""
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logging.info(f"PDF '{uploaded_file.name}' saved successfully.")
        return file_path
    except Exception as e:
        logging.error(f"Failed to save PDF: {str(e)}")
        return None

@traceable(run_type="llm", metadata={"ls_provider": "ollama", "model": "mistral"})
def create_qa_agent(pdf_paths, model_name="mistral"):
    """Create or load the question-answering agent for multiple PDFs."""
    vectorstore = None
    for pdf_path in pdf_paths:
        logging.info(f"Processing PDF: {pdf_path}")
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300, length_function=len)
        splits = text_splitter.split_documents(pages)
        
        embeddings = OllamaEmbeddings(model=model_name)
        
        if vectorstore is None:
            vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        
        for chunk in tqdm(splits, desc=f"Processing chunks in {pdf_path}"):
            vectorstore.add_documents([chunk], embedding=embeddings)

    llm = Ollama(model=model_name)

    prompt_template = """
    You are a helpful AI assistant that answers questions based on the provided PDF document.
    Use only the context provided to answer the question. If you don't know the answer, say so.
    
    Context: {context}
    
    Question: {question}
    
    Answer: """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 50}), 
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

@traceable(run_type="chain")
def ask_question(qa_chain, question):
    """Ask the question to the QA agent and return the result."""
    try:
        response = qa_chain({"query": question})
        return {
            "answer": response["result"],
            "sources": [doc.page_content for doc in response["source_documents"]]
        }
    except Exception as e:
        logging.error(f"An error occurred while asking the question: {str(e)}")
        return {"error": str(e), "answer": None, "sources": None}

st.title("Chat with PDF using Niki AI")

user_input = st.text_area("Enter your question:")

col1, col2 = st.columns([4, 1])

with col1:
    if st.button("Ask"):
        if not user_input.strip():
            st.warning("Please enter a question before asking.")
        elif not pdf_exists():
            st.warning("Please upload a PDF to proceed.")
        else:
            if "qa_agent" not in st.session_state:
                st.session_state["qa_agent"] = create_qa_agent(st.session_state["pdf_paths"], model_name="mistral")
            
            result = ask_question(st.session_state["qa_agent"], user_input)
            if result.get("error"):
                st.error(result["error"])
            else:
                st.subheader("Niki AI Response:")
                st.write(result["answer"])

with col2:
    uploaded_files = st.file_uploader("Choose PDFs to upload", type=["pdf"], key="pdf_uploader", label_visibility="collapsed", accept_multiple_files=True)
    if uploaded_files:
        st.session_state["pdf_paths"] = []
        for uploaded_file in uploaded_files:
            pdf_path = save_uploaded_file(uploaded_file)
            if pdf_path:
                st.session_state["pdf_paths"].append(pdf_path)
                st.success(f"PDF '{uploaded_file.name}' uploaded successfully. Processing...")

        if "qa_agent" in st.session_state:
            del st.session_state["qa_agent"]  
        if "pdf_paths" in st.session_state and st.session_state["pdf_paths"]:
            st.session_state["qa_agent"] = create_qa_agent(st.session_state["pdf_paths"], model_name="mistral")