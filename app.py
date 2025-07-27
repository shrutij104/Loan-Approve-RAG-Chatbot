import streamlit as st
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
from langchain.chains import RetrievalQA
import os

st.title("💰 Loan Approval RAG Chatbot")
st.write("Ask questions about loan applications and their approval status based on the provided dataset.")

# --- Configuration ---
# Ensure you have your dataset in the same directory or provide the full path
DATASET_PATH = 'Training Dataset.csv'
HUGGINGFACE_MODEL_ID = "google/flan-t5-base" # You can try "google/flan-t5-small" or other light models

# --- Caching Data Loading and Model Loading ---
@st.cache_resource
def load_data_and_create_vectorstore():
    try:
        df = pd.read_csv(DATASET_PATH)
        st.success("Dataset loaded successfully!")
    except FileNotFoundError:
        st.error(f"Error: {DATASET_PATH} not found. Please upload it or ensure it's in the correct directory.")
        st.stop() # Stop execution if data isn't found

    documents = []
    for index, row in df.iterrows():
        doc_content = (
            f"Loan Application ID: {row['Loan_ID']}. "
            f"Gender: {row['Gender']}, Married: {row['Married']}, Dependents: {row['Dependents']}, "
            f"Education: {row['Education']}, Self Employed: {row['Self_Employed']}, "
            f"Applicant Income: {row['ApplicantIncome']}, Coapplicant Income: {row['CoapplicantIncome']}, "
            f"Loan Amount: {row['LoanAmount']}, Loan Amount Term: {row['Loan_Amount_Term']}, "
            f"Credit History: {row['Credit_History']}, Property Area: {row['Property_Area']}. "
            f"Loan Status: {row['Loan_Status']}."
        )
        documents.append(Document(page_content=doc_content, metadata={"row_id": index, "Loan_ID": row['Loan_ID']}))

    # No extensive chunking for this dataset, each row is a "chunk"
    chunks = documents

    st.info("Creating embeddings and building vector store. This might take a moment...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.success("Vector store created!")
    return vectorstore

@st.cache_resource
def load_llm():
    try:
        tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(HUGGINGFACE_MODEL_ID)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            device=0 if torch.cuda.is_available() else -1
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        st.success(f"Loaded Hugging Face LLM: {HUGGINGFACE_MODEL_ID}")
        return llm
    except Exception as e:
        st.error(f"Could not load Hugging Face model {HUGGINGFACE_MODEL_ID}. Error: {e}")
        st.warning("Consider using a cloud LLM (OpenAI, Gemini, etc.) if you have API keys and credits.")
        return None # Return None if LLM loading fails

# Load resources
vectorstore = load_data_and_create_vectorstore()
llm = load_llm()

if vectorstore and llm:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about loan approvals..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                full_response = response['result']
                source_docs = response['source_documents']

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(full_response)
                    st.markdown("**--- Source Documents ---**")
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Document {i+1} (ID: {doc.metadata.get('Loan_ID', 'N/A')}):**")
                        st.markdown(f"```\n{doc.page_content}\n```")

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response + "\n\n--- Source Documents ---\n" + "\n".join([f"Document {i+1} (ID: {doc.metadata.get('Loan_ID', 'N/A')}): {doc.page_content}" for i, doc in enumerate(source_docs)])})

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})
else:
    st.warning("Please resolve the issues above to start the chatbot.")