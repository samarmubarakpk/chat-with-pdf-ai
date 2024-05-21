import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = google_api_key

st.title("Gemma Model Document Q&A")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

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
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        loader = PyPDFDirectoryLoader("./encoder_decoder")  # Data Ingestion
        docs = loader.load()  # Document Loading
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        final_documents = text_splitter.split_documents(docs[:20])  # Splitting
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
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    except Exception as e:
        st.error(f"Error in retrieving documents: {e}")
