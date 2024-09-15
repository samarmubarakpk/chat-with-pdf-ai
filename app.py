import streamlit as st
import os
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directly assign the OpenAI API key
openai_api_key = ''
os.environ["OPENAI_API_KEY"] = openai_api_key

st.title("One Identity installation help")

# Initialize the language model with OpenAI
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

@st.cache_resource(show_spinner=False)
def vector_embedding():
    try:
        # Use OpenAI Embeddings instead of Google embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        loader = PyPDFDirectoryLoader("./encoder_decoder")  # Data Ingestion
        docs = loader.load()  # Document Loading
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        final_documents = text_splitter.split_documents(docs[:150])  # Splitting
        vectors = FAISS.from_documents(final_documents, embeddings)  # Vector embeddings
        return vectors
    except Exception as e:
        st.error(f"Error in vector embedding process: {e}")
        raise

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    try:
        st.session_state.vectors = vector_embedding()
        st.write("Vector Store DB Is Ready")
    except Exception as e:
        st.error(f"Error in creating vector store DB: {e}")

if prompt1 and "vectors" in st.session_state:
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time: ", time.process_time() - start)
        st.write(response['answer'])

        # With a Streamlit expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    except Exception as e:
        st.error(f"Error in retrieving documents: {e}")
