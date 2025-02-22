import streamlit as st
import os
import time
import tempfile
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from PIL import Image
import io
import re
import base64
import hashlib
from dotenv import load_dotenv
import anthropic
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Q&A with Vector Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'pdf_images' not in st.session_state:
    st.session_state.pdf_images = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None
if 'chunk_mapping' not in st.session_state:
    st.session_state.chunk_mapping = {}
if 'token_usage' not in st.session_state:
    st.session_state.token_usage = 0


# At the start of your app, add this function
def load_existing_vector_database():
    """Try to load existing vector database if available"""
    try:
        if os.path.exists("vector_db/pdf_index.faiss") and os.path.exists("vector_db/chunk_mapping.pkl"):
            # Load FAISS index
            index = faiss.read_index("vector_db/pdf_index.faiss")
            
            # Load chunk mapping
            with open("vector_db/chunk_mapping.pkl", "rb") as f:
                chunk_mapping = pickle.load(f)
            
            # Store in session state
            st.session_state.vector_index = index
            st.session_state.chunk_mapping = chunk_mapping
            
            # Initialize embedding model
            if not st.session_state.embedding_model:
                initialize_embedding_model()
            
            return True
    except Exception as e:
        st.error(f"Error loading vector database: {str(e)}")
    return False


def initialize_embedding_model():
    """Initialize the sentence transformer model for embeddings"""
    if st.session_state.embedding_model is None:
        with st.spinner("Loading embedding model..."):
            # Use a smaller model for efficiency
            st.session_state.embedding_model = SentenceTransformer('all-mpnet-base-v2')
            # Financial/business oriented model
            # st.session_state.embedding_model = SentenceTransformer('finbert-finance-sentiment') 
            # st.session_state.embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')  # Better for Q&A




            return True
    return True


def extract_text_and_images(pdf_files, chunk_size=500, chunk_overlap=200):
    """Extract text and images from PDFs and create chunks with metadata"""
    all_text = ""
    all_images = {}
    chunks = []
    chunk_metadata = []
    image_hashes = set()
    
    for pdf_idx, pdf_file in enumerate(pdf_files):
        pdf_num = pdf_idx + 1
        doc = None
        pdf_path = None
        
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                pdf_path = tmp_file.name
            
            # Extract images with PyMuPDF
            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                st.error(f"Error opening PDF {pdf_num} with PyMuPDF: {str(e)}")
                continue
                
            try:
                pdf_reader = PdfReader(pdf_path)
            except Exception as e:
                st.error(f"Error opening PDF {pdf_num} with PyPDF2: {str(e)}")
                continue
            
            for page_idx, page in enumerate(pdf_reader.pages):
                page_num = page_idx + 1
                
                try:
                    # Extract text
                    page_text = page.extract_text() or ""
                except Exception as e:
                    st.warning(f"Could not extract text from PDF {pdf_num}, Page {page_num}: {str(e)}")
                    page_text = ""
                
                # Add to complete text (for display purposes)
                if page_text:
                    all_text += f"\n\n--- PDF {pdf_num}, PAGE {page_num} ---\n{page_text}"
                
                # Process images on this page
                page_images = []
                try:
                    pymupdf_page = doc.load_page(page_idx)
                    
                    # Get images
                    image_list = pymupdf_page.get_images(full=True)
                    for img_idx, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Check for duplicates using hash
                            img_hash = hashlib.md5(image_bytes).hexdigest()
                            if img_hash in image_hashes:
                                continue
                            
                            image_hashes.add(img_hash)
                            image_key = f"pdf_{pdf_num}_page_{page_num}_img_{img_idx+1}"
                            all_images[image_key] = image_bytes
                            page_images.append(image_key)
                        except Exception as e:
                            st.warning(f"Failed to extract image {img_idx+1} from PDF {pdf_num}, Page {page_num}: {str(e)}")
                    
                    # If no embedded images, try to render page as image with error handling
                    if not image_list:
                        try:
                            # Increase DPI and add white background
                            zoom = 2  # higher zoom for better quality
                            mat = fitz.Matrix(zoom, zoom)
                            pix = pymupdf_page.get_pixmap(matrix=mat, alpha=False)
                            
                            if pix.width > 0 and pix.height > 0 and len(pix.samples) > 0:
                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                img_byte_arr = io.BytesIO()
                                img.save(img_byte_arr, format='JPEG', quality=85)
                                rendered_bytes = img_byte_arr.getvalue()
                                
                                if len(rendered_bytes) > 0:  # Verify we got valid image data
                                    render_hash = hashlib.md5(rendered_bytes).hexdigest()
                                    if render_hash not in image_hashes:
                                        image_hashes.add(render_hash)
                                        image_key = f"pdf_{pdf_num}_page_{page_num}_full_render"
                                        all_images[image_key] = rendered_bytes
                                        page_images.append(image_key)
                            else:
                                st.warning(f"Page {page_num} of PDF {pdf_num} produced empty pixmap")
                        except Exception as e:
                            st.warning(f"Failed to render page {page_num} as image for PDF {pdf_num}: {str(e)}")
                            
                except Exception as e:
                    st.warning(f"Error processing page {page_num} of PDF {pdf_num}: {str(e)}")
                    continue
                
                # Create text chunks with overlap
                if page_text:
                    words = page_text.split()
                    
                    for chunk_start in range(0, len(words), chunk_size - chunk_overlap):
                        chunk_end = min(chunk_start + chunk_size, len(words))
                        if chunk_end - chunk_start < 50:  # Skip tiny chunks
                            continue
                            
                        chunk_text = ' '.join(words[chunk_start:chunk_end])
                        chunk_id = len(chunks)
                        chunks.append(chunk_text)
                        
                        chunk_metadata.append({
                            'id': chunk_id,
                            'pdf_num': pdf_num,
                            'pdf_name': pdf_file.name,
                            'page_num': page_num,
                            'chunk_start': chunk_start,
                            'chunk_end': chunk_end,
                            'images': page_images.copy()
                        })
        
        finally:
            # Clean up resources
            if doc:
                try:
                    doc.close()
                except:
                    pass
                    
            # Try to remove temporary file with retry
            if pdf_path:
                for _ in range(3):  # Try up to 3 times
                    try:
                        os.unlink(pdf_path)
                        break
                    except Exception:
                        time.sleep(0.1)  # Wait briefly before retry
    
    return all_text, all_images, chunks, chunk_metadata

def create_vector_database(chunks, chunk_metadata):
    """Create FAISS vector database from text chunks"""
    if not chunks:
        st.error("No text chunks found to index")
        return None, {}
    
    # Ensure embedding model is initialized
    if not st.session_state.embedding_model:
        if not initialize_embedding_model():
            st.error("Failed to initialize embedding model")
            return None, {}
    
    # Compute embeddings in batches
    batch_size = 32
    all_embeddings = []
    
    with st.spinner(f"Creating embeddings for {len(chunks)} chunks..."):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:min(i+batch_size, len(chunks))]
            batch_embeddings = st.session_state.embedding_model.encode(batch)
            all_embeddings.extend(batch_embeddings)
    
    # Create FAISS index
    embedding_dim = len(all_embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Convert to numpy array and add to index
    embeddings_array = np.array(all_embeddings).astype('float32')
    index.add(embeddings_array)
    
    # Create mapping dictionary
    chunk_mapping = {i: {"text": chunks[i], "metadata": chunk_metadata[i]} 
                     for i in range(len(chunks))}
    
    # Save to disk
    save_dir = "vector_db"
    os.makedirs(save_dir, exist_ok=True)
    
    faiss.write_index(index, f"{save_dir}/pdf_index.faiss")
    with open(f"{save_dir}/chunk_mapping.pkl", "wb") as f:
        pickle.dump(chunk_mapping, f)
    
    return index, chunk_mapping

def semantic_search(query, k=3):
    """Search for most relevant chunks using vector similarity"""
    if not st.session_state.vector_index or not st.session_state.embedding_model:
        st.error("Vector database not initialized")
        return []
    
    # Encode query
    query_vector = st.session_state.embedding_model.encode([query]).astype('float32')
    
    # Search
    distances, indices = st.session_state.vector_index.search(query_vector, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0:  # FAISS may return -1 for not enough results
            continue
            
        idx = int(idx)
        chunk_info = st.session_state.chunk_mapping.get(idx)
        if not chunk_info:
            continue
            
        results.append({
            "text": chunk_info["text"],
            "metadata": chunk_info["metadata"],
            "score": float(distances[0][i])
        })
    
    return results

def query_claude_with_retrieval(query, model="claude-3-sonnet-20240229"):
    """Query Claude API with semantically retrieved context"""
    # Get Anthropic client
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        return "Error: Anthropic API key not configured"
        
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    
    try:
        # Perform semantic search
        results = semantic_search(query, k=5)
        
        if not results:
            return "I couldn't find relevant information in the provided documents. Please try a different question or upload documents with the information you're looking for."
        
        # Build context from search results
        context_parts = []
        image_references = set()
        
        for result in results:
            # Add text chunk
            meta = result["metadata"]
            context_part = f"--- PDF {meta['pdf_num']}, PAGE {meta['page_num']} ---\n{result['text']}"
            context_parts.append(context_part)
            
            # Collect image references
            for img_ref in meta["images"]:
                image_references.add(img_ref)
        
        # Combine context
        context = "\n\n".join(context_parts)
        
        # Add image references if available
        if image_references:
            image_section = "\n\nRelevant images:\n" + "\n".join(
                f"[Image {ref.split('_')[1]}.{ref.split('_')[3]}.{ref.split('_')[5]}]" 
                if "img" in ref else f"[Image {ref.split('_')[1]}.{ref.split('_')[3]}.full]"
                for ref in image_references
            )
            context += image_section
        
        # System prompt
        system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided PDF content.
If the answer isn't in the provided context, say 'I don't find information about this in the provided PDFs.'

IMPORTANT IMAGE HANDLING INSTRUCTIONS:
1. When referring to images or charts, use the format "[Image X.Y.Z]" where X=pdf number, Y=page number, Z=image number.
   Example: Use "[Image 1.2.3]" instead of "pdf_1_page_2_img_3"

2. When the user asks about ANY visual element (chart, graph, figure, table, etc.), always reference relevant images 
   using the [Image X.Y.Z] format.

Give comprehensive, accurate answers based only on the PDF context."""
        
        # Query Claude
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Context from PDFs:\n\n{context}\n\nQuestion: {query}"}
            ]
        )
        
        # Update token usage estimate
        input_tokens = len(context.split()) * 1.3
        output_tokens = len(response.content[0].text.split()) * 1.3
        st.session_state.token_usage += int(input_tokens + output_tokens)
        
        return response.content[0].text
    except Exception as e:
        return f"Error querying Claude: {str(e)}"

def display_image(img_ref):
    """Display an image from the session state"""
    if img_ref in st.session_state.pdf_images:
        try:
            img_bytes = st.session_state.pdf_images[img_ref]
            image = Image.open(io.BytesIO(img_bytes))
            
            # Extract components for caption
            parts = re.match(r'pdf_(\d+)_page_(\d+)_(img_(\d+)|full_render)', img_ref)
            if parts:
                pdf_num, page_num = parts.group(1), parts.group(2)
                is_full_page = 'full_render' in img_ref
                img_num = parts.group(4) if not is_full_page else 'full'
                caption = f"Image {pdf_num}.{page_num}.{img_num}"
            else:
                caption = img_ref
                
            st.image(image, caption=caption, use_column_width=True)
            return True
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
            return False
    else:
        st.warning(f"Image not found: {img_ref}")
        return False

def extract_image_references(response_text):
    """Extract all image references from Claude's response"""
    images = []
    
    # Look for [Image X.Y.Z] format
    matches = re.findall(r'\[Image (\d+)\.(\d+)\.(\d+|\w+)\]', response_text)
    for pdf_num, page_num, img_num in matches:
        if img_num.lower() == 'full':
            ref = f"pdf_{pdf_num}_page_{page_num}_full_render"
        else:
            ref = f"pdf_{pdf_num}_page_{page_num}_img_{img_num}"
            
        if ref in st.session_state.pdf_images and ref not in images:
            images.append(ref)
    
    return images

# Title
st.title("📚 PDF Q&A with Vector Search")

if st.button("Check Local Database"):
    if os.path.exists("vector_db/pdf_index.faiss"):
        file_size = os.path.getsize("vector_db/pdf_index.faiss") / (1024 * 1024)  # Size in MB
        st.success(f"✅ Vector database found locally! Size: {file_size:.2f} MB")
        st.info(f"Location: {os.path.abspath('vector_db/pdf_index.faiss')}")
    else:
        st.error("❌ No local vector database found")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        anthropic_api_key = st.text_input("Enter Anthropic API Key", type="password")
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
            st.success("✅ API key set")
    else:
        st.success("✅ API key loaded from environment")
    
    # Model selection
    model = st.selectbox(
        "Select Claude Model",
        [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229", 
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307"
        ],
        index=0
    )
    
    # PDF upload
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process PDFs"):
            if not initialize_embedding_model():
                st.error("Failed to initialize embedding model")
            else:
                with st.spinner("Processing PDFs..."):
                    # Extract text and images
                    all_text, all_images, chunks, chunk_metadata = extract_text_and_images(uploaded_files)
                    
                    # Store in session state
                    st.session_state.extracted_text = all_text
                    st.session_state.pdf_images = all_images
                    
                    # Create vector database
                    index, mapping = create_vector_database(chunks, chunk_metadata)
                    
                    if index and mapping:
                        st.session_state.vector_index = index
                        st.session_state.chunk_mapping = mapping
                        st.success(f"✅ Processed {len(uploaded_files)} PDF(s) with {len(chunks)} chunks")
                    else:
                        st.error("Failed to create vector database")
    
    # Show stats
    if st.session_state.vector_index:
        st.metric("Vector DB Size", f"{st.session_state.vector_index.ntotal} chunks")
    
    if st.session_state.pdf_images:
        st.metric("Images Extracted", len(st.session_state.pdf_images))
    
    st.metric("Estimated Token Usage", st.session_state.token_usage)

# Main chat interface
# if st.session_state.vector_index and st.session_state.pdf_images:
#     # Show all images button
#     if st.button("Show All Images"):
#         st.header("All Extracted Images")
        
#         # Create a multi-column layout
#         cols = st.columns(2)
#         for i, (img_ref, _) in enumerate(st.session_state.pdf_images.items()):
#             with cols[i % 2]:
#                 display_image(img_ref)
    
#     # Chat history
#     for q, a in st.session_state.chat_history:
#         with st.chat_message("user"):
#             st.write(q)
#         with st.chat_message("assistant"):
#             st.write(a)
            
#             # Extract and display referenced images
#             image_refs = extract_image_references(a)
#             if image_refs:
#                 st.write("---")
#                 for ref in image_refs:
#                     display_image(ref)
    
#     # Query input
#     query = st.chat_input("Ask a question about your documents:")
#     if query:
#         # Display user message
#         with st.chat_message("user"):
#             st.write(query)
        
#         # Generate and display response
#         with st.chat_message("assistant"):
#             response_container = st.empty()
#             with st.spinner("Generating answer..."):
#                 response = query_claude_with_retrieval(query, model=model)
#                 response_container.write(response)
                
#                 # Extract and display referenced images
#                 image_refs = extract_image_references(response)
#                 if image_refs:
#                     st.write("---")
#                     st.subheader("Referenced Images")
                    
#                     # Create columns for images
#                     img_cols = st.columns(min(len(image_refs), 2))
#                     for i, ref in enumerate(image_refs):
#                         with img_cols[i % 2]:
#                             display_image(ref)
            
#             # Update chat history
#             st.session_state.chat_history.append((query, response))
# else:
#     st.info("👈 Please upload and process PDF files using the sidebar to get started.")


if not st.session_state.vector_index:
    if load_existing_vector_database():
        st.success("✅ Loaded existing vector database!")
        # Show stats
        st.metric("Vector DB Size", f"{st.session_state.vector_index.ntotal} chunks")
        st.info("You can start asking questions or upload new PDFs to add to the database")
    else:
        st.info("👈 Please upload PDF files using the sidebar to get started.")

# # Main chat interface - only show if we have a vector index
# if st.session_state.vector_index:
#     # Chat history
#     for q, a in st.session_state.chat_history:
#         with st.chat_message("user"):
#             st.write(q)
#         with st.chat_message("assistant"):
#             st.write(a)
            
#     # Query input
#     query = st.chat_input("Ask a question about your documents:")
#     if query:
#         # Display user message
#         with st.chat_message("user"):
#             st.write(query)
        
#         # Generate and display response
#         with st.chat_message("assistant"):
#             response_container = st.empty()
#             with st.spinner("Generating answer..."):
#                 response = query_claude_with_retrieval(query, model=model)
#                 response_container.write(response)
                
#                 # Update chat history
#                 st.session_state.chat_history.append((query, response))

# Main chat interface - only show if we have a vector index
if st.session_state.vector_index:
    # Chat history
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            
            # Add image display for chat history
            image_refs = extract_image_references(a)
            if image_refs:
                st.write("---")
                st.subheader("Referenced Images")
                cols = st.columns(min(len(image_refs), 2))
                for i, ref in enumerate(image_refs):
                    with cols[i % 2]:
                        display_image(ref)
            
    # Query input
    query = st.chat_input("Ask a question about your documents:")
    if query:
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Generate and display response
        with st.chat_message("assistant"):
            response_container = st.empty()
            with st.spinner("Generating answer..."):
                response = query_claude_with_retrieval(query, model=model)
                response_container.write(response)
                
                # Display images referenced in the response
                image_refs = extract_image_references(response)
                if image_refs:
                    st.write("---")
                    st.subheader("Referenced Images")
                    cols = st.columns(min(len(image_refs), 2))
                    for i, ref in enumerate(image_refs):
                        with cols[i % 2]:
                            display_image(ref)
                
                # Update chat history
                st.session_state.chat_history.append((query, response))


# Add instructions
with st.expander("How to use this app"):
    st.markdown("""
    ### Instructions
    1. Enter your Anthropic API key if not set in environment variables
    2. Upload one or more PDF files
    3. Click 'Process PDFs' to extract text and create vector database
    4. Ask questions about the content using the chat interface
    
    ### Benefits of Vector Search
    - Much more cost-effective for large documents
    - Only sends relevant context to the LLM
    - Faster responses
    - Works with local embedding model
    
    ### Tips
    - Be specific in your questions
    - Reference specific charts by saying "chart" or "figure"
    - The app works best with searchable PDFs, not scanned documents
    """)