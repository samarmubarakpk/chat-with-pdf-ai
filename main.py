import streamlit as st
import os
import time
import pyttsx3
from deep_translator import GoogleTranslator
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Load environment variables
load_dotenv()

# Directly assign the OpenAI API key
openai_api_key = ''
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize the language model with OpenAI
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4")
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")



# Initialize translator for multi-language support using deep_translator
def translate_text(text, target_language):
    try:
        translator = GoogleTranslator(source='auto', target=target_language)
        return translator.translate(text)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text  # Fallback to original text if translation fails

# # Set up text-to-speech


engine = pyttsx3.init(driverName="espeak")


# Streamlit UI setup
st.title("Interactive PDF Chat with Multi-language and Voice Support")

# Sidebar for PDF management
uploaded_files = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True, type=['pdf'])
if uploaded_files:
    st.sidebar.write("Uploaded PDFs:")
    for file in uploaded_files:
        st.sidebar.write(f"- {file.name}")

# Language selection
language = st.selectbox("Select Language for Response", ["English", "Spanish", "French", "German", "Chinese"])

# Prompt setup
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
def vector_embedding(files):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        all_docs = []
        for uploaded_file in files:
            # Save the uploaded file locally
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())
            
            # Load the saved file with PyPDFLoader
            loader = PyPDFLoader(uploaded_file.name)  # Data ingestion
            docs = loader.load()
            all_docs.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
        final_documents = text_splitter.split_documents(all_docs)  # Splitting
        vectors = FAISS.from_documents(final_documents, embeddings)  # Vector embeddings
        return vectors
    except Exception as e:
        st.error(f"Error in vector embedding process: {e}")
        raise


if st.button("Embed Uploaded PDFs"):
    try:
        st.session_state.vectors = vector_embedding([file for file in uploaded_files])
        st.write("Vector Store DB is Ready")
    except Exception as e:
        st.error(f"Error in creating vector store DB: {e}")

# Input for user question
prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1 and "vectors" in st.session_state:
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time: ", time.process_time() - start)
        
        # Translate the response if a non-English language is selected
        response_text = response['answer']
        if language != "English":
            lang_code = {
                "Spanish": "es",
                "French": "fr",
                "German": "de",
                "Chinese": "zh"
            }.get(language, "en")  # Default to English if not mapped
            response_text = translate_text(response_text, lang_code)
        st.write(response_text)
        
        # Text-to-speech for the response
        engine.say(response_text)
        engine.runAndWait()

        # Show document similarity search
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    except Exception as e:
        st.error(f"Error in retrieving documents: {e}")
