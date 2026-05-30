import streamlit as st
import pandas as pd
from io import BytesIO
from pypdf import PdfReader

# Local summarization libraries
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Download required text processing data locally
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_data()

# Try importing OCR, handle gracefully if still installing
try:
    import pytesseract
    from pdf2image import convert_from_bytes
except ImportError:
    pass

# ==========================================
# 1. Page Configuration & Custom RTL CSS
# ==========================================
st.set_page_config(page_title="100% Local AI Summarizer", layout="wide")

st.markdown("""
    <style>
    .arabic-text { text-align: right; direction: rtl; font-family: 'Cairo', sans-serif; font-size: 1.1rem; }
    .english-text { text-align: left; direction: ltr; font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Local Text Extraction & Summarization Logic
# ==========================================
def extract_pdf_text(file_bytes, mode):
    text = ""
    reader = PdfReader(BytesIO(file_bytes))
    
    # Extract digital text layers
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
            
    # Fallback or explicit handwriting OCR mode
    if mode == "Handwritten / Scanned Image" or not text.strip():
        try:
            st.info("Reading handwriting/images using local OCR engine...")
            images = convert_from_bytes(file_bytes)
            ocr_text = []
            for img in images:
                # Custom config triggers both English and Arabic local packages
                page_data = pytesseract.image_to_string(img, lang='eng+ara')
                ocr_text.append(page_data)
            text = "\n".join(ocr_text)
        except Exception as e:
            if mode == "Handwritten / Scanned Image":
                st.error("OCR Engine is initializing on the cloud. Try standard mode or re-uploading in a moment.")
    return text

def extract_tabular_data(file_bytes, extension):
    if extension == 'csv':
        df = pd.read_csv(BytesIO(file_bytes))
    else:
        df = pd.read_excel(BytesIO(file_bytes))
    return df.to_string(index=False), df

def generate_local_summary(text, length, lang_code):
    """Summarizes text mathematically using Latent Semantic Analysis (LSA)"""
    parser = PlaintextParser.from_string(text, Tokenizer(lang_code))
    summarizer = LsaSummarizer()
    
    # Calculate target sentence count based on length choice
    total_sentences = len(text.split('.'))
    if length == "Short":
        sentence_count = max(2, int(total_sentences * 0.05))
    elif length == "Medium":
        sentence_count = max(5, int(total_sentences * 0.15))
    else:
        sentence_count = max(10, int(total_sentences * 0.30))
        
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])

# ==========================================
# 3. User Interface Layout
# ==========================================
st.title("📄 100% Offline-Style Cloud Summarizer")
st.write("No APIs, no keys, no timeouts. Processing is handled fully inside the app.")

st.sidebar.header("Configuration / الإعدادات")
language = st.sidebar.selectbox("Language / لغة الملف الأساسية", ["English", "العربية"])
length_choice = st.sidebar.selectbox(
    "Summary Detail / حجم الملخص", 
    ["Short (قصير)", "Medium (متوسط)", "Long (طويل)"]
)

pdf_mode = st.sidebar.radio(
    "PDF Processing Mode",
    ["Standard Digital Text", "Handwritten / Scanned Image"]
)

uploaded_file = st.file_uploader(
    "Drop your document here (PDF, CSV, XLSX)", 
    type=["pdf", "csv", "xlsx", "xls"]
)

# ==========================================
# 4. Processing Pipeline
# ==========================================
if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    extracted_context = ""
    dataframe_preview = None
    raw_bytes = uploaded_file.read()
    
    with st.spinner("Extracting content locally..."):
        if file_ext == 'pdf':
            extracted_context = extract_pdf_text(raw_bytes, pdf_mode)
        elif file_ext in ['csv', 'xlsx', 'xls']:
            extracted_context, dataframe_preview = extract_tabular_data(raw_bytes, file_ext)

    if dataframe_preview is not None:
        with st.expander("📊 Preview Spreadsheet Data"):
            st.dataframe(dataframe_preview.head(10))
    elif extracted_context.strip():
        with st.expander("📝 View Extracted Text Preview"):
            st.text_area("Context Stream", value=extracted_context[:1500], height=150, disabled=True)

    if st.button("✨ Generate Summary / إنشاء الملخص", type="primary"):
        if not extracted_context.strip():
            st.error("No readable text found. If it's handwriting, ensure you selected 'Handwritten / Scanned Image' in the sidebar.")
        else:
            with st.spinner("Calculating summary..."):
                lang_code = "arabic" if language == "العربية" else "english"
                summary_output = generate_local_summary(extracted_context, length_choice.split()[0], lang_code)
                
                st.success("Summary Ready!")
                if language == "العربية":
                    st.markdown(f'<div class="arabic-text"><h3>الملخص:</h3><p>{summary_output}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="english-text"><h3>Summary:</h3><p>{summary_output}</p></div>', unsafe_allow_html=True)
