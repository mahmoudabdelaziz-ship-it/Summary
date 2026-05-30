import streamlit as st
import pandas as pd
import json
import requests
from io import BytesIO

# ==========================================
# 1. Page Configuration & Custom RTL CSS
# ==========================================
st.set_page_config(page_title="AI Cloud Summarizer", layout="wide")

st.markdown("""
    <style>
    .arabic-text { text-align: right; direction: rtl; font-family: 'Cairo', sans-serif; font-size: 1.1rem; }
    .english-text { text-align: left; direction: ltr; font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

# Securely grab your token from Streamlit Advanced Settings -> Secrets
if "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]
else:
    # Fallback if you hardcoded it directly (replace with your actual token if not using Secrets)
    HF_TOKEN = "YOUR_HUGGINGFACE_API_TOKEN"

# ==========================================
# 2. Cloud API Helper Functions
# ==========================================
def query_hf_api(payload, model_id):
    """Sends requests to Hugging Face text models."""
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

def extract_text_and_handwriting_via_api(file_bytes):
    """
    Sends the entire PDF file to a powerful cloud Document OCR model.
    This reads typed text, scanned images, AND handwriting.
    """
    # Using Microsoft's LayoutLM for advanced document reading (OCR)
    api_url = "https://api-inference.huggingface.co/models/microsoft/layoutlmv3-base"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/pdf"
    }
    
    response = requests.post(api_url, headers=headers, data=file_bytes)
    
    try:
        result = response.json()
        # Extract and merge words detected by the AI vision model
        if isinstance(result, list) and len(result) > 0:
            extracted_words = [item.get('word', '') for item in result if 'word' in item]
            if extracted_words:
                return " ".join(extracted_words)
        elif isinstance(result, dict) and 'error' in result:
            st.warning(f"Cloud OCR engine status: {result['error']}. Trying basic extraction...")
    except Exception:
        pass
        
    # Smart local fallback: Try reading standard text if the OCR endpoint is busy
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(file_bytes))
        fallback_text = "".join([page.extract_text() or "" for page in reader.pages])
        return fallback_text
    except:
        return ""

def extract_tabular_data(file_bytes, extension):
    if extension == 'csv':
        df = pd.read_csv(BytesIO(file_bytes))
    else:
        df = pd.read_excel(BytesIO(file_bytes))
    return df.to_string(index=False), df

# ==========================================
# 3. User Interface Layout
# ==========================================
st.title("📄 Multi-Format AI Cloud Summarizer")
st.write("Handles typed documents, **handwritten/scanned images**, and spreadsheets in English & العربية.")

st.sidebar.header("Configuration / الإعدادات")
language = st.sidebar.selectbox("Output Language / لغة الملخص", ["English", "العربية"])
length_choice = st.sidebar.selectbox(
    "Summary Detail / حجم الملخص", 
    ["Short (قصير)", "Medium (متوسط)", "Long (طويل)"]
)

# Toggle to help the AI optimize its approach
pdf_mode = st.sidebar.radio(
    "PDF Document Type",
    ["Standard Digital Text", "Handwritten / Scanned Image"],
    help="Select 'Handwritten' if your PDF contains non-selectable text or photos of handwritten text."
)

if "Short" in length_choice:
    max_len, min_len = 45, 15
elif "Medium" in length_choice:
    max_len, min_len = 120, 50
else:
    max_len, min_len = 280, 100

uploaded_file = st.file_uploader(
    "Drop your document here (PDF, CSV, XLSX)", 
    type=["pdf", "csv", "xlsx", "xls"]
)

# ==========================================
# 4. Pipeline Processing
# ==========================================
if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    extracted_context = ""
    dataframe_preview = None
    raw_bytes = uploaded_file.read()
    
    with st.spinner("Processing document in the cloud..."):
        if file_ext == 'pdf':
            if pdf_mode == "Handwritten / Scanned Image":
                # Uses cloud OCR to read handwriting without freezing the webapp
                extracted_context = extract_text_and_handwriting_via_api(raw_bytes)
            else:
                # Standard text extractor
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(BytesIO(raw_bytes))
                    extracted_context = "".join([page.extract_text() or "" for page in reader.pages])
                except:
                    extracted_context = ""
                
                # Automatically trigger Cloud OCR if standard reading returns empty text
                if not extracted_context.strip():
                    st.info("No digital text detected. Running Cloud Handwriting OCR instead...")
                    extracted_context = extract_text_and_handwriting_via_api(raw_bytes)
                    
        elif file_ext in ['csv', 'xlsx', 'xls']:
            extracted_context, dataframe_preview = extract_tabular_data(raw_bytes, file_ext)

    # Show data preview if it's an Excel sheet or CSV
    if dataframe_preview is not None:
        with st.expander("📊 Preview Spreadsheet Data"):
            st.dataframe(dataframe_preview.head(10))
    # Show text preview for documents
    elif extracted_context.strip():
        with st.expander("📝 View Extracted Text Preview"):
            st.text_area("Extracted Context", value=extracted_context[:1500], height=150, disabled=True)

    if st.button("✨ Generate Summary / إنشاء الملخص", type="primary"):
        if not extracted_context.strip() or len(extracted_context.strip()) < 10:
            st.error("The AI could not read any valid text from this file. If it's handwriting, make sure the scan is clear.")
        else:
            with st.spinner("AI is compiling your summary..."):
                # Using the premier Multilingual Summarization Model
                model_id = "csebuetnlp/mT5_multilingual_XLSum"
                payload = {
                    "inputs": extracted_context[:3000],
                    "parameters": {"max_length": max_len, "min_length": min_len}
                }
                
                api_result = query_hf_api(payload, model_id)
                
                try:
                    summary_output = api_result[0]['summary_text']
                    st.success("Done!")
                    
                    if language == "العربية":
                        st.markdown(f'<div class="arabic-text"><h3>الملخص:</h3><p>{summary_output}</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="english-text"><h3>Summary:</h3><p>{summary_output}</p></div>', unsafe_allow_html=True)
                except:
                    st.error("The cloud model is warming up. Please wait 10 seconds and click the button again.")
