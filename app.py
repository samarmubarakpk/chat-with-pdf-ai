import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import base64
from PIL import Image
import io
import re
from dotenv import load_dotenv
import anthropic
import hashlib

# Set page configuration - MUST COME FIRST
st.set_page_config(
    page_title="PDF Q&A Assistant with Claude",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Set up Anthropic API key
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = None

# Create Session State variables
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'pdf_images' not in st.session_state:
    st.session_state.pdf_images = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main App Title
st.title("📚 PDF Q&A Assistant with Claude")

# Now handle the API key in sidebar
with st.sidebar:
    st.header("Upload PDFs")
    
    # Request API key if not in environment
    if not anthropic_api_key:
        anthropic_api_key = st.text_input("Enter your Anthropic API Key", type="password")

    # Initialize Anthropic client
    if anthropic_api_key:
        try:
            anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
            st.success("✅ Anthropic API key loaded successfully")
        except Exception as e:
            st.error(f"Error initializing Anthropic client: {str(e)}")
    else:
        st.error("❌ Anthropic API key not configured or invalid")

# The rest of your application functions and UI elements...
def extract_images_and_text(pdf_files):
    """Extract both text and images from PDF files with improved image deduplication"""
    all_text = ""
    all_images = {}
    image_hashes = {}  # Store image hashes to detect duplicates
    
    for i, pdf_file in enumerate(pdf_files):
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                pdf_path = tmp_file.name
            
            # Extract text with PyPDF2
            pdf_reader = PdfReader(pdf_path)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    all_text += f"\n\n--- PDF {i+1}, PAGE {page_num+1} ---\n{page_text}"
            
            # Enhanced image extraction with PyMuPDF
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Method 1: Extract normal images with deduplication
                image_list = page.get_images(full=True)
                has_embedded_images = False
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Hash the image for deduplication
                    img_hash = hashlib.md5(image_bytes).hexdigest()
                    
                    # Check if this image is a duplicate
                    if img_hash in image_hashes:
                        # Image already stored - add a reference to text
                        original_key = image_hashes[img_hash]
                        ref_text = f"\n\n[IMAGE_REF: {original_key}] Cross-reference to another image\n\n"
                        all_text += ref_text
                        continue
                    
                    # Create a normalized identifier for each image
                    image_key = f"pdf_{i+1}_page_{page_num+1}_img_{img_index+1}"
                    image_hashes[img_hash] = image_key
                    all_images[image_key] = image_bytes
                    has_embedded_images = True
                    
                    # Add descriptive image reference to text
                    surrounding_text = page_text[max(0, len(page_text)//2 - 200):min(len(page_text), len(page_text)//2 + 200)]
                    image_description = get_image_description(surrounding_text)
                    
                    ref_text = f"\n\n[IMAGE: {image_key}] {image_description}\n\n"
                    all_text += ref_text
                
                # Method 2: Only render page as image if no embedded images found
                if not has_embedded_images:
                    # If no images found, render the entire page as an image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    image_bytes = pix.tobytes()
                    
                    # Convert to PIL Image and then to bytes in a common format
                    img = Image.frombytes("RGB", [pix.width, pix.height], image_bytes)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    rendered_bytes = img_byte_arr.getvalue()
                    
                    # Hash the rendered page for deduplication
                    render_hash = hashlib.md5(rendered_bytes).hexdigest()
                    
                    # Check if this rendered page is a duplicate
                    if render_hash not in image_hashes:
                        # Create a unique identifier
                        image_key = f"pdf_{i+1}_page_{page_num+1}_full_render"
                        image_hashes[render_hash] = image_key
                        all_images[image_key] = rendered_bytes
                        
                        # Add reference to text
                        ref_text = f"\n\n[IMAGE: {image_key}] Full page render\n\n"
                        all_text += ref_text
            
            # Clean up temporary file
            os.unlink(pdf_path)
            
        except Exception as e:
            st.error(f"Error processing PDF {i+1}: {str(e)}")
    
    return all_text, all_images

def get_image_description(surrounding_text):
    """Generate image description based on surrounding text"""
    surrounding_text = surrounding_text.lower()
    
    # Look for chart types
    if any(term in surrounding_text for term in ["chart", "price", "trend", "candlestick"]):
        return "CHART: Price/Trend visualization"
    elif any(term in surrounding_text for term in ["momentum", "rsi", "macd", "stochastic"]):
        return "INDICATOR: Momentum/Technical indicator"
    elif any(term in surrounding_text for term in ["table", "summary", "comparison"]):
        return "TABLE: Data summary"
    elif any(term in surrounding_text for term in ["recommendation", "target", "buy", "sell"]):
        return "RECOMMENDATION: Investment advice"
    else:
        return "Financial visualization"

def normalize_image_reference(reference):
    """Normalize image references to a standard format"""
    # Handle [Image X.Y.Z] format
    new_style_match = re.match(r'\[Image (\d+)\.(\d+)\.(\d+)\]', reference)
    if new_style_match:
        pdf_num, page_num, img_num = new_style_match.groups()
        return f"pdf_{pdf_num}_page_{page_num}_img_{img_num}"
    
    # Handle pdf_X_page_Y_img_Z format
    trad_match = re.match(r'pdf_(\d+)_page_(\d+)_img_(\d+)', reference)
    if trad_match:
        return reference  # Already in normalized format
    
    # Handle full render format
    render_match = re.match(r'pdf_(\d+)_page_(\d+)_full_render', reference)
    if render_match:
        return reference  # Already in normalized format
    
    return None  # Invalid reference

def track_displayed_image(image_ref):
    """Track that an image has been displayed in this session"""
    if 'displayed_images_history' not in st.session_state:
        st.session_state.displayed_images_history = set()
    
    # Normalize the reference before tracking
    normalized_ref = normalize_image_reference(image_ref)
    if normalized_ref:
        st.session_state.displayed_images_history.add(normalized_ref)

def was_image_displayed(image_ref):
    """Check if an image was already displayed in this session"""
    if 'displayed_images_history' not in st.session_state:
        return False
    
    # Normalize the reference before checking
    normalized_ref = normalize_image_reference(image_ref)
    if normalized_ref:
        return normalized_ref in st.session_state.displayed_images_history
    
    return False

def display_image(img_ref, context=None, image_type=None, force=False):
    """Unified function to display images with tracking and deduplication"""
    # Skip if already displayed and not forced
    if was_image_displayed(img_ref) and not force:
        return False
    
    # Normalize the reference
    normalized_ref = normalize_image_reference(img_ref)
    if not normalized_ref:
        st.warning(f"Invalid image reference format: {img_ref}")
        return False
    
    if normalized_ref in st.session_state.pdf_images:
        img_bytes = st.session_state.pdf_images[normalized_ref]
        try:
            image = Image.open(io.BytesIO(img_bytes))
            
            # Determine caption based on image type and reference
            caption = determine_image_caption(normalized_ref, image_type, context)
            
            # Display the image
            st.image(image, caption=caption, use_container_width=True)
            
            # Track that this image has been displayed
            track_displayed_image(normalized_ref)
            return True
            
        except Exception as e:
            st.error(f"Error displaying image {normalized_ref}: {str(e)}")
            return False
    else:
        st.warning(f"Image not found: {normalized_ref}")
        return False

def determine_image_caption(img_ref, image_type=None, context=None):
    """Generate appropriate caption for images"""
    # Extract components from reference
    components = re.match(r'pdf_(\d+)_page_(\d+)_(img_(\d+)|full_render)', img_ref)
    
    if not components:
        return f"Image: {img_ref}"
    
    pdf_num, page_num = components.group(1), components.group(2)
    is_full_render = components.group(3) == "full_render"
    img_num = components.group(4) if not is_full_render else None
    
    # Format basic reference info
    ref_text = f"Image {pdf_num}.{page_num}" + (f".{img_num}" if img_num else " (Full Page)")
    
    # If image type is provided, use it
    if image_type:
        if image_type == "price_chart":
            return f"Technical Price Chart: {ref_text}"
        elif image_type == "momentum":
            return f"Momentum Indicator: {ref_text}"
        elif image_type == "recommendation":
            return f"Recommendation Summary: {ref_text}"
        elif image_type == "table":
            return f"Data Table: {ref_text}"
        else:
            return f"{image_type.capitalize()}: {ref_text}"
    
    # Otherwise try to infer from reference and context
    if context:
        context_lower = context.lower()
        if "price" in context_lower and "chart" in context_lower:
            return f"Technical Price Chart: {ref_text}"
        elif any(term in context_lower for term in ["momentum", "rsi", "macd"]):
            return f"Technical Indicator: {ref_text}" 
        elif any(term in context_lower for term in ["recommendation", "summary", "target"]):
            return f"Recommendation Summary: {ref_text}"
            
    # Default caption based on location in document
    if is_full_render:
        return f"Full Page Render: {ref_text}"
    elif page_num == "1":
        return f"Overview: {ref_text}"
    elif page_num == "2" and img_num == "1":
        return f"Price Chart: {ref_text}"
    elif page_num == "2" and img_num == "2":
        return f"Technical Indicator: {ref_text}"
    elif page_num == "3" and img_num == "1":
        return f"Recommendation: {ref_text}"
    else:
        return f"Visualization: {ref_text}"

def extract_image_references_from_response(response_text):
    """Extract all image references from Claude's response"""
    references = []
    
    # Look for [Image X.Y.Z] format
    new_style_refs = re.findall(r'\[Image (\d+)\.(\d+)\.(\d+)\]', response_text)
    for pdf_num, page_num, img_num in new_style_refs:
        normalized_ref = f"pdf_{pdf_num}_page_{page_num}_img_{img_num}"
        if normalized_ref not in references:
            references.append(normalized_ref)
    
    # Look for pdf_X_page_Y_img_Z format  
    traditional_refs = re.findall(r'pdf_(\d+)_page_(\d+)_img_(\d+)', response_text)
    for pdf_num, page_num, img_num in traditional_refs:
        normalized_ref = f"pdf_{pdf_num}_page_{page_num}_img_{img_num}"
        if normalized_ref not in references:
            references.append(normalized_ref)
            
    # Look for full page renders
    render_refs = re.findall(r'pdf_(\d+)_page_(\d+)_full_render', response_text)
    for pdf_num, page_num in render_refs:
        normalized_ref = f"pdf_{pdf_num}_page_{page_num}_full_render"
        if normalized_ref not in references:
            references.append(normalized_ref)
            
    return references

def find_relevant_images(query, available_images):
    """Find images relevant to a query based on content/location heuristics"""
    query_lower = query.lower()
    relevant_images = []
    
    # Check for specific image requests
    if re.search(r'(image|chart|graph|figure|table|visualization)\s+(\d+)\.(\d+)\.(\d+)', query_lower):
        # User is asking for specific image by number
        matches = re.findall(r'(image|chart|graph|figure|table|visualization)\s+(\d+)\.(\d+)\.(\d+)', query_lower)
        for _, pdf_num, page_num, img_num in matches:
            img_ref = f"pdf_{pdf_num}_page_{page_num}_img_{img_num}"
            if img_ref in available_images and img_ref not in relevant_images:
                relevant_images.append(img_ref)
    
    # Content-based selection for charts
    is_price_chart_query = any(term in query_lower for term in 
                              ["price chart", "price action", "candlestick", "technical chart"])
    is_momentum_query = any(term in query_lower for term in 
                          ["momentum", "rsi", "macd", "oscillator", "technical indicator"])
    is_recommendation_query = any(term in query_lower for term in 
                                ["recommendation", "rating", "buy signal", "sell signal", "target price"])
    
    # If searching for charts, find relevant ones
    if is_price_chart_query:
        # Price charts are typically early in document
        price_charts = [img for img in available_images.keys() 
                      if re.match(r'pdf_\d+_page_[1-3]_img_\d+', img)]
        relevant_images.extend([img for img in price_charts if img not in relevant_images])
        
    if is_momentum_query:
        # Look for momentum charts (typically page 2, after price chart)
        momentum_charts = [img for img in available_images.keys() 
                         if re.match(r'pdf_\d+_page_2_img_[2-4]', img)]
        relevant_images.extend([img for img in momentum_charts if img not in relevant_images])
        
    if is_recommendation_query:
        # Recommendation charts often near end of document
        rec_charts = [img for img in available_images.keys() 
                    if re.match(r'pdf_\d+_page_[3-5]_img_\d+', img)]
        relevant_images.extend([img for img in rec_charts if img not in relevant_images])
    
    return relevant_images

# Main UI logic for handling query responses with images
def handle_query_response(query, response, images):
    """Handle display of images based on query and response"""
    # Extract all image references from response
    referenced_images = extract_image_references_from_response(response)
    
    # Find relevant images based on query content
    relevant_images = find_relevant_images(query, images)
    
    # Combine and deduplicate while maintaining priority (referenced first)
    all_images_to_display = []
    for img in referenced_images + relevant_images:
        if img not in all_images_to_display and img in images:
            all_images_to_display.append(img)
    
    # Display images if found
    if all_images_to_display:
        st.write("---")
        st.subheader("Relevant Visualizations")
        
        # Use columns for layout
        if len(all_images_to_display) > 1:
            cols = st.columns(min(len(all_images_to_display), 2))
            for i, img_ref in enumerate(all_images_to_display):
                with cols[i % 2]:
                    # Determine image type based on position and query
                    image_type = None
                    if "price" in query.lower() and i == 0:
                        image_type = "price_chart"
                    elif "momentum" in query.lower() and i <= 1:
                        image_type = "momentum"
                    
                    display_image(img_ref, response, image_type)
        else:
            # Single image - use full width
            display_image(all_images_to_display[0], response)
    
    return all_images_to_display

def split_text_into_chunks(text, max_tokens=12000):
    """Split text into chunks of roughly equal size"""
    # Simple splitting by paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # Rough token estimate (words / 0.75)
        paragraph_tokens = len(paragraph.split()) / 0.75
        
        if len(current_chunk.split()) / 0.75 + paragraph_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def query_claude(query, context, model="claude-3-sonnet-20240229"):
    """Query Claude API with the query and context"""
    if not anthropic_client:
        return "Error: Anthropic client not initialized. Please check your API key."
    
    try:
        system_prompt = """You are a helpful financial analysis assistant that answers questions based ONLY on the provided PDF content.
If the answer isn't in the provided context, say 'I don't find information about this in the provided PDFs.'

IMPORTANT IMAGE HANDLING INSTRUCTIONS FOR FINANCIAL REPORTS:
1. When referring to images or charts, DO NOT use the format "pdf_X_page_Y_img_Z". 
   Instead, use the format "[Image X.Y.Z]" where X=pdf number, Y=page number, Z=image number.
   Example: Use "[Image 1.2.3]" instead of "pdf_1_page_2_img_3"

2. For technical analysis reports, actively identify and reference images that contain:
   - Price charts (often have candlesticks, trend lines, moving averages)
   - Momentum indicators (RSI, MACD, etc.)
   - Performance tables
   - Recommendation summaries
   - Technical indicator visualizations

3. When the user asks about ANY visual element (using terms like "chart", "graph", "figure", "table", "image", 
   "visualization", "indicator", "momentum score", "price action", "technical analysis"), explicitly list ALL 
   relevant image references in your response using the [Image X.Y.Z] format.

4. ESPECIALLY IMPORTANT: For any question related to price charts, technical analysis, or indicators, ALWAYS 
   mention the specific image references where these can be visualized, even if the user doesn't explicitly ask for images.

Example responses specifically for financial reports:
- "The technical price chart for 4imprint Group is in [Image 1.2.1]. As shown in this chart, the price is above both the 21-day and 50-day moving averages."
- "I can see several important visualizations in this report: [Image 1.1.1] (main price chart), [Image 1.2.1] (momentum indicator), and [Image 1.3.1] (recommendation summary table)."
- "The RSI reading of 72.29 is mentioned in the text and visualized in the technical chart [Image 1.2.2]."
- "According to the analysis in the PDF and the chart in [Image 1.2.1], 4imprint Group is currently trading above its descending trendline."

Give comprehensive, accurate answers based only on the PDF context. When in doubt about a chart or table reference, always include the image reference so the user can see it."""
        
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Context from PDFs:\n\n{context}\n\nQuestion: {query}"}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error querying Claude: {str(e)}"

# Continue with the sidebar UI
with st.sidebar:
    # File uploader that accepts multiple PDFs
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files (up to 500MB each)",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process PDFs"):
            with st.spinner("Extracting text and images from PDFs..."):
                # Use the enhanced extraction function
                extracted_text, pdf_images = extract_images_and_text(uploaded_files)
                
                st.session_state.extracted_text = extracted_text
                st.session_state.pdf_images = pdf_images
                st.session_state.chat_history = []
                st.success(f"Successfully processed {len(uploaded_files)} PDF(s)")

    # Model selection
    model = st.selectbox(
        "Select Claude Model",
        [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229", 
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307"
        ],
        index=0  # Default to 3.5 Sonnet
    )
    
    # Add token consumption tracker
    if 'token_usage' not in st.session_state:
        st.session_state.token_usage = 0
        
    st.metric("Estimated Token Usage", st.session_state.token_usage)
    
    # Show stats
    if st.session_state.extracted_text:
        word_count = len(st.session_state.extracted_text.split())
        st.info(f"Extracted {word_count} words and {len(st.session_state.pdf_images)} images")
        st.text(f"Estimated tokens: ~{int(word_count * 1.3)}")

# Main chat interface
if st.session_state.extracted_text:
    st.subheader("Ask questions about your PDFs")
    
    # Add image gallery button
    if st.button("Show All Images"):
        if st.session_state.pdf_images:
            st.subheader("All Images in Documents")
            cols = st.columns(2)
            for i, (img_ref, img_bytes) in enumerate(st.session_state.pdf_images.items()):
                try:
                    with cols[i % 2]:
                        # Use the new display_image function
                        display_image(img_ref, force=True)
                except Exception as e:
                    st.error(f"Could not display image {img_ref}: {str(e)}")
        else:
            st.info("No images found in the processed documents.")
    
    # Display chat history
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
    
    # User query input with chat input
    query = st.chat_input("Ask a question about your PDFs:")
    
    if query:
        # Add user question to chat history display
        with st.chat_message("user"):
            st.write(query)
            
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...")
            
            with st.spinner("Generating answer..."):
                # Split the text into manageable chunks
                chunks = split_text_into_chunks(st.session_state.extracted_text)
                
                # For small documents, just query once
                if len(chunks) == 1:
                    final_response = query_claude(query, chunks[0], model=model)
                    # Estimate token usage
                    input_tokens = len(chunks[0].split()) * 1.3
                    output_tokens = len(final_response.split()) * 1.3
                    st.session_state.token_usage += int(input_tokens + output_tokens)
                else:
                    # Query each chunk and combine responses
                    all_responses = []
                    input_tokens = 0
                    
                    for chunk in chunks:
                        response = query_claude(query, chunk, model=model)
                        all_responses.append(response)
                        input_tokens += len(chunk.split()) * 1.3
                    
                    # Combine responses 
                    synthesis_prompt = f"""
                    Based on the previous partial analyses of the PDF content, provide a comprehensive 
                    answer to the question: '{query}'
                    
                    Previous analyses:
                    {'-' * 40}
                    {''.join(all_responses)}
                    {'-' * 40}
                    
                    Synthesize a complete answer that captures all relevant information from the documents.
                    """
                    
                    final_response = query_claude(
                        synthesis_prompt, 
                        "",  # No additional context needed
                        model=model
                    )
                    
                    # Estimate token usage
                    output_tokens = len(final_response.split()) * 1.3
                    synthesis_tokens = len(synthesis_prompt.split()) * 1.3
                    st.session_state.token_usage += int(input_tokens + output_tokens + synthesis_tokens)
                
                # Process Claude's response to find image references
                # First look for [Image X.Y.Z] style references
                new_style_pattern = r'\[Image (\d+)\.(\d+)\.(\d+)\]'
                new_style_matches = re.findall(new_style_pattern, final_response)
                
                # Also look for traditional pdf_X_page_Y_img_Z references
                traditional_pattern = r'pdf_(\d+)_page_(\d+)_img_(\d+)'
                traditional_matches = re.findall(traditional_pattern, final_response)
                
                # Combine all unique references
                all_references = []
                for match in new_style_matches + traditional_matches:
                    pdf_num, page_num, img_num = match
                    img_ref = f"pdf_{pdf_num}_page_{page_num}_img_{img_num}"
                    if img_ref not in all_references and img_ref in st.session_state.pdf_images:
                        all_references.append(img_ref)
                
                # Display Claude's response
                response_placeholder.markdown(final_response)
                
                # If we found image references, display buttons
                if all_references:
                    st.write("---")
                    st.subheader("Referenced Images")
                    
                    # Create column layout for buttons
                    cols = st.columns(min(3, len(all_references)))
                    
                    # Create buttons for each reference
                    for i, img_ref in enumerate(all_references):
                        # Extract parts from the reference for display
                        parts = re.match(r'pdf_(\d+)_page_(\d+)_img_(\d+)', img_ref)
                        if parts:
                            pdf_num, page_num, img_num = parts.groups()
                            
                            # Create button in the appropriate column
                            with cols[i % len(cols)]:
                                button_key = f"show_image_{img_ref}_{i}"
                                if st.button(f"Show Image {pdf_num}.{page_num}.{img_num}", key=button_key):
                                    # Set selected image and rerun
                                    st.session_state.selected_image = img_ref
                                    st.experimental_rerun()
                
                # Display selected image if any
                if 'selected_image' in st.session_state:
                    selected_ref = st.session_state.selected_image
                    if selected_ref in st.session_state.pdf_images:
                        st.write("---")
                        st.subheader(f"Viewing: {selected_ref}")
                        
                        # Use the new display_image function with force=True
                        if display_image(selected_ref, force=True):
                            # Add button to hide image
                            if st.button("Hide Image", key=f"hide_{selected_ref}"):
                                del st.session_state.selected_image
                                st.experimental_rerun()
                
                # Use the new consolidated handling function
                handle_query_response(query, final_response, st.session_state.pdf_images)
            
            # Add to chat history
            st.session_state.chat_history.append((query, final_response))
else:
    st.info("👈 Please upload PDF files using the sidebar to get started.")

# Add enhanced instructions
with st.expander("Help & Information"):
    st.markdown("""
    ### How to use this app:
    1. Enter your Anthropic API key in the sidebar (if not set in environment variables)
    2. Upload one or more PDF files using the sidebar
    3. Click 'Process PDFs' to extract text and images
    4. Ask questions about the content of your PDFs using the chat interface
    5. For images/graphs, try queries like "Show me the graph about..." or "What does figure X show?"
    
    ### Features:
    - Processes both text and images from your PDFs
    - Maintains chat history for context
    - Supports all Claude 3 models
    - Tracks estimated token usage
    - Handles large documents by splitting into chunks
    
    ### Tips:
    - Be specific in your questions
    - For multi-page PDFs, you can ask about specific pages
    - The app works best with searchable PDFs (not scanned documents)
    - Complex tables and charts may have limited interpretation
    """)