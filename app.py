import streamlit as st
import pandas as pd
from io import BytesIO
from pypdf import PdfReader

# Local mathematical text summarization libraries
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Download required text tokenizer data locally on startup
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_data()

# Initialize OCR libraries safely
try:
    import pytesseract
    from pdf2image import convert_from_bytes
except ImportError:
    pass

# ==========================================
# 1. Page Configuration & Custom RTL CSS
# ==========================================
st.set_page_config(page_title="Universal Local AI Summarizer", layout="wide")

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
    try:
        reader = PdfReader(BytesIO(file_bytes))
        # Step 1: Always check for a digital text layer first
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading digital layers: {e}")
            
    # Step 2: Fallback or explicit handwriting OCR mode if text is blank
    if mode == "Handwritten / Scanned Image" or not text.strip():
        try:
            st.info("Reading handwriting/images using local OCR engine...")
            images = convert_from_bytes(file_bytes)
            ocr_text = []
            for img in images:
                # Triggers both the English and Arabic system language packs installed via packages.txt
                page_data = pytesseract.image_to_string(img, lang='eng+ara')
                ocr_text.append(page_data)
            text = "\n".join(ocr_text)
        except Exception as e:
            if mode == "Handwritten / Scanned Image":
                st.error("The system OCR engine is currently setting up on the cloud instance. Please try re-uploading in a moment.")
    return text

def extract_tabular_data(file_bytes, extension):
    if extension == 'csv':
        df = pd.read_csv(BytesIO(file_bytes))
    else:
        df = pd.read_excel(BytesIO(file_bytes))
    return df.to_string(index=False), df

def generate_local_summary(text, length, lang_code):
    """Clean watermark noise and summarize mathematically using Latent Semantic Analysis (LSA)"""
    # 1. Filter out common scanner app watermark artifacts (like CamScanner)
    watermarks = ["camscanner", "scanned with", "scanned by", "برنامج كام سكانر", "كام سكانر"]
    cleaned_lines = []
    
    for line in text.split('\n'):
        # Skip lines that are just short chunks of watermark spam
        if any(wm in line.lower() for wm in watermarks) and len(line.strip()) < 25:
            continue
        # Strip background watermark words entirely from longer sentences
        for wm in watermarks:
            line = line.replace(wm, "").replace(wm.upper(), "").replace(wm.capitalize(), "")
        cleaned_lines.append(line)
        
    cleaned_text = "\n".join(cleaned_lines).strip()
    
    # Validation boundary check
    if not cleaned_text or len(cleaned_text.split()) < 4:
        if lang_code == "arabic":
            return "الملخص غير متاح: المستند يحتوي على علامات مائية فقط أو أن جودة خط اليد غير واضحة للقرائة."
        else:
            return "Summary unavailable: The document contains only watermark spam or the handwriting is too faint to parse."

    # 2. Process text into mathematical sentence vectors
    parser = PlaintextParser.from_string(cleaned_text, Tokenizer(lang_code))
    summarizer = LsaSummarizer()
    
    total_sentences = max(1, len(cleaned_text.split('.')))
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
st.write("Process typed files, handwritten pages, and spreadsheets cleanly without external APIs.")

st.sidebar.header("Configuration / الإعدادات")
language = st.sidebar.selectbox("Language / لغة الملف الأساسية", ["English", "العربية"])
length_choice = st.sidebar.selectbox(
    "Summary Detail / حجم الملخص", 
    ["Short (قصير)", "Medium (متوسط)", "Long (طويل)"]
)

pdf_mode = st.sidebar.radio(
    "PDF Processing Mode",
    ["Standard Digital Text", "Handwritten / Scanned Image"],
    help="Switch to Handwritten mode if the document consists of notebook scans or unselectable image text."
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
    
    with st.spinner("Extracting content segments locally..."):
        if file_ext == 'pdf':
            extracted_context = extract_pdf_text(raw_bytes, pdf_mode)
        elif file_ext in ['csv', 'xlsx', 'xls']:
            extracted_context, dataframe_preview = extract_tabular_data(raw_bytes, file_ext)

    # UI Previews
    if dataframe_preview is not None:
        with st.expander("📊 Preview Spreadsheet Data"):
            st.dataframe(dataframe_preview.head(10))
    elif extracted_context.strip():
        with st.expander("📝 View Extracted Text Preview"):
            st.text_area("Parsed Text Data Stream", value=extracted_context[:1500], height=150, disabled=True)

    if st.button("✨ Generate Summary / إنشاء الملخص", type="primary"):
        if not extracted_context.strip():
            st.error("No text found. If this is a handwriting scan, double-check that you selected 'Handwritten / Scanned Image' in the sidebar configuration.")
        else:
            with st.spinner("Calculating summary matrix..."):
                lang_code = "arabic" if language == "العربية" else "english"
                summary_output = generate_local_summary(extracted_context, length_choice.split()[0], lang_code)
                
                st.success("Summary Ready!")
                if language == "العربية":
                    st.markdown(f'<div class="arabic-text"><h3>الملخص المستنتج:</h3><p>{summary_output}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="english-text"><h3>Document Summary:</h3><p>{summary_output}</p></div>', unsafe_allow_html=True)
