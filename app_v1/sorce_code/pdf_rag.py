import os
import streamlit as st
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
import torch
from streamlit_modal import Modal
from streamlit_pdf_viewer import pdf_viewer
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor
from openai import OpenAI
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed
from auth_config import run_authentication
from openai import OpenAI
import time
import streamlit as st

if "login_success" not in st.session_state:
    st.session_state["login_success"] = False

USER_DB = {
    "alice": "123",
    "bob": "456",
    "jam": "789"
}


def login():
    st.title("🔐 User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_DB and USER_DB[username] == password:
            st.session_state["login_success"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("❌ error")


# login to judge the branch
if not st.session_state["login_success"]:
    login()
    st.stop()

st.write(f"Welcome，{st.session_state['username']}！")

if st.button("Logout"):
    st.session_state.clear()
    st.rerun()

# ---------------------- Static Settings ----------------------
st.set_page_config(layout="wide")

st.title("📚 Document Research Assistance System")

st.markdown("""
<style>
    .appview-container .main .block-container {{
        padding-top: {padding_top}rem;
        padding-bottom: {padding_bottom}rem;
        padding-left: {padding_left}%;
        padding-right: {padding_right}%;
        }}
</style>""".format(
    padding_top=1, padding_bottom=1, padding_left=10, padding_right=10
),
    unsafe_allow_html=True,
)  # Page margin settings

HF_CACHE = "/root/autodl-tmp/colqwen2-base"
if not st.session_state.get("login_success"):
    login()
    st.stop()

username = st.session_state["username"]
PDF_DIR = f"chat-with-pdf/pdfs/{username}/"
IMG_DIR = f"chat-with-pdf/images/{username}/"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

client = OpenAI(
    api_key="f9e888a6-ae48-4ff8-80ba-16165c683753",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# ---------------------- Session State Initialization ----------------------
for key, default in {
    'uploaded_files_query': None,  # File uploader component return value (multiple files)
    'file_path_query': None,  # List of saved file paths (multiple files)
    'pdf_submitted_query': False,  # Submission flag for Query module
    'review_button_query': False,  # Whether preview is triggered in Query module
    'query_processing_done': False,  # Whether image extraction is completed
    'query': None,  # Current input query
    'all_images': [],  # List of extracted images

    'uploaded_file_chat': None,  # File uploader component return value (single file)
    'file_path_chat': None,  # Saved file path (single file)
    'pdf_submitted_chat': False,  # Submission flag for Chat module
    'review_button_chat': False,  # Whether preview is triggered in Chat module
    'chat_history': [],  # Chat history records
    'pending_question': None  # Question waiting to be answered
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------- PDF Upload Controllers ----------------------

def _save_single_file(file, progress_callback=None):
    """Internal helper: Save single file with chunked writing and progress callback"""
    try:
        # Ensure target directory exists
        os.makedirs(PDF_DIR, exist_ok=True)

        path = os.path.join(PDF_DIR, file.name)
        file.seek(0, 2)  # Move to end of file
        file_size = file.tell()  # Get total file size
        file.seek(0)  # Reset file pointer

        with open(path, "wb", buffering=1024 * 1024) as f:  # 1MB buffer
            bytes_written = 0
            for chunk in iter(lambda: file.read(8192), b""):  # Read in 8KB chunks
                f.write(chunk)
                bytes_written += len(chunk)
                if progress_callback:
                    progress_callback(bytes_written / file_size)
        return path
    except Exception as e:
        st.error(f"Failed to save file {file.name}: {str(e)}")
        return None
    finally:
        file.seek(0)  # Reset file pointer for subsequent use


def upload_single_pdf(file):
    """Upload single PDF file with progress bar"""
    if file is None:
        return None

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress):
        progress_bar.progress(progress)
        status_text.text(f"Upload progress: {int(progress * 100)}%")

    try:
        result = _save_single_file(file, update_progress)
        if result:
            st.success(f"File {file.name} uploaded successfully!")
        return result
    finally:
        progress_bar.empty()
        status_text.empty()


def upload_multiple_pdfs(files):
    """Upload multiple PDF files with parallel processing"""
    if not files:
        return []

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(files)

    try:
        paths = []
        completed = 0

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Create task dictionary (future: file)
            futures = {executor.submit(_save_single_file, file): file for file in files}

            # Process results in completion order
            for future in as_completed(futures):
                path = future.result()
                paths.append(path)
                completed += 1
                progress_bar.progress(completed / total_files)
                status_text.text(f"Completed {completed}/{total_files} files")

        # Count success/failure
        successful = [p for p in paths if p is not None]
        if len(successful) != total_files:
            st.warning(f"Successfully uploaded {len(successful)}/{total_files} files")
        else:
            st.success(f"All {total_files} files uploaded successfully!")

        return successful

    except Exception as e:
        st.error(f"Error during upload: {str(e)}")
        return []
    finally:
        progress_bar.empty()
        status_text.empty()


# ---------------------- PDF Viewer ----------------------
def show_pdf_modal(files, modal_key):
    modal = Modal(key=modal_key, title="📑 Preview Uploaded PDF", max_width=850)
    with modal.container():
        for file in files:
            st.write(Path(file).name)
            pdf_viewer(input=file, width=750)


# ---------------------- PDF to Image ----------------------
def extract_images_from_pdfs(file_paths):
    all_images = []
    for file_path in file_paths:
        folder = os.path.join(IMG_DIR, Path(file_path).stem)
        os.makedirs(folder, exist_ok=True)
        if not os.listdir(folder):  # Avoid multiple transition
            pages = convert_from_path(file_path)
            for i, page in enumerate(pages):
                img_path = os.path.join(folder, f"page_{i + 1}.png")
                page.save(img_path, "PNG")
        for img_file in sorted(Path(folder).glob("*.png")):
            all_images.append(Image.open(img_file).convert("RGB"))
    return all_images


# ---------------------- Loading ColQwen2 Module ----------------------
def load_colqwen2():
    if "col_model" not in st.session_state:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        st.session_state.col_model = ColQwen2.from_pretrained(
            HF_CACHE,
            torch_dtype="auto",
            device_map="cuda:0",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        st.session_state.processor = ColQwen2Processor.from_pretrained(HF_CACHE)


# ---------------------- Query Function ----------------------
def handle_query(query):
    if 'col_model' not in st.session_state or 'processor' not in st.session_state:
        raise ValueError("Model or processor not initialized in session state")

    if 'all_images' not in st.session_state or not st.session_state.all_images:
        raise ValueError("No images available for querying")

    model = st.session_state.col_model
    processor = st.session_state.processor
    all_images = st.session_state.all_images
    batch_size = 4

    batch_queries = processor.process_queries([query])
    batch_queries = {k: v.to(model.device) for k, v in batch_queries.items()}

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i + batch_size]
            inputs = processor.process_images(batch)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            out = model(**inputs)
            embeddings.append(out)
        image_embeddings = torch.cat(embeddings, dim=0)
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    top_idx = scores[0].argsort(descending=True)[:3]
    return [(all_images[i.item()], scores[0][i].item()) for i in top_idx]


# ---------------------- Chat LLM Processing ----------------------
def sambanova_chat(messages):
    try:
        response = client.chat.completions.create(
            model="deepseek-r1-250528",
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Chat Error] {str(e)}"


# ---------------------- Chat / File Loading ----------------------
def load_pdf(path):
    return PDFPlumberLoader(path).load()


def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def index_docs(docs):
    st.session_state.pdf_text = "\n\n".join([d.page_content for d in docs])


def retrieve_docs(_):
    return st.session_state.get("pdf_text", "[ERROR] No document loaded")


def answer_question(q, ctx):
    prompt = f"""
You are a professional assistant. Answer based on this document. If unsure, say: "Not found in the document."

Document:
{ctx}

Question:
{q}
"""
    return sambanova_chat([
        {"role": "system", "content": "You answer PDF-related questions."},
        {"role": "user", "content": prompt}
    ])


# ---------------------- Layout：2 Columns ----------------------
mainCol1, mainCol2 = st.columns(2)

with mainCol1:
    st.header("🔍 PDF Query")

    # Initialization
    if 'uploaded_files_query' not in st.session_state:
        st.session_state['uploaded_files_query'] = []

    # Query upload multiple files
    uploaded_files_query = st.file_uploader(
        "Upload one or more PDFs", type="pdf", accept_multiple_files=True, key="uploader_query",
    )
    if uploaded_files_query:
        st.session_state.uploaded_files_query = uploaded_files_query
        st.session_state.file_path_query = upload_multiple_pdfs(uploaded_files_query)

    btcol1, btcol2, _ = st.columns([2, 2, 1])
    if st.session_state.file_path_query:
        with btcol1:
            if st.button("Preview Uploaded PDFs", key="btn_review_query"):
                st.session_state.review_button_query = True

    # if st.button("Submit Query PDF", key="btn_submit_query") and st.session_state.file_path_query:
    #     st.session_state.pdf_submitted_query = True
    #     st.rerun()
    if st.session_state.file_path_query:
        with btcol2:
            if st.button("Submit Query PDF", key="btn_submit_query"):
                st.session_state.pdf_submitted_query = True
                st.rerun()

if st.session_state.review_button_query:
    # Indent the show modal block to display popup in the page center
    show_pdf_modal(st.session_state.file_path_query, "popup_query")
    st.session_state.review_button_query = False

with mainCol1:
    if st.session_state.pdf_submitted_query and not st.session_state.query_processing_done:
        st.write("📄 Processing PDF for query...")
        st.session_state.all_images = extract_images_from_pdfs(st.session_state.file_path_query)
        st.session_state.query_processing_done = True
        st.success("PDF ready. Enter your query below.")

    if st.session_state.query_processing_done:
        query = st.chat_input("Enter your query about the uploaded PDFs:", key="query_input")

        if query:
            st.session_state.query = query
            st.write(f"💬 You asked: _{query}_")
            load_colqwen2()
            with st.spinner("Processing your query..."):
                results = handle_query(query)
                st.session_state.query_results = results
                if results[0][1] < 1.0:
                    st.warning("❌ No strong match found in the PDFs.")
                else:
                    for img, score in results:
                        st.image(img, caption=f"Similarity: {score:.2f}")

        # Show the latest result when no query is input
        # Avoid user mis-click causing query disappear
        elif st.session_state.get("query_results"):
            st.write(f"💬 You asked: _{st.session_state.query}_")
            for img, score in st.session_state.query_results:
                st.image(img, caption=f"Similarity: {score:.2f}")

with mainCol2:
    st.header("💬 Chat with PDF")

    st.session_state.file_path_chat = None

    uploaded_file_chat = st.file_uploader(
        "Upload a single PDF", type="pdf", accept_multiple_files=False, key="uploader_chat"
    )
    if uploaded_file_chat and not st.session_state.file_path_chat:
        try:
            with st.spinner("Uploading PDF..."):
                st.session_state.uploaded_file_chat = uploaded_file_chat
                st.session_state.file_path_chat = upload_single_pdf(uploaded_file_chat)


        except Exception as e:
            st.error(f"Upload failed: {str(e)}")
            st.session_state.file_path_chat = None

        #

    #         st.session_state.uploaded_file_chat = uploaded_file_chat
    #         st.session_state.file_path_chat = upload_single_pdf(uploaded_file_chat)

    btcol3, btcol4, _ = st.columns([2, 2, 1])
    if st.session_state.file_path_chat:
        with btcol3:
            if st.button("Preview Uploaded PDF", key="btn_review_chat"):
                st.session_state.review_button_chat = True
    # if st.button("Submit Chat PDF", key="btn_submit_chat") and st.session_state.file_path_chat:
    #     st.session_state.pdf_submitted_chat = True
    #     st.rerun()
    if st.session_state.file_path_chat:
        with btcol4:
            if st.button("Submit Chat PDF", key="btn_submit_chat"):
                st.session_state.pdf_submitted_chat = True
                st.rerun()

if st.session_state.review_button_chat:
    # Indent the show modal block to display popup in the page center
    show_pdf_modal([st.session_state.file_path_chat], "popup_chat")
    st.session_state.review_button_chat = False

with mainCol2:
    if st.session_state.pdf_submitted_chat and st.session_state.file_path_chat:
        st.write("🧾 Indexing PDF for chat...")
        with st.spinner("Parsing and indexing..."):
            docs = load_pdf(st.session_state.file_path_chat)
            index_docs(split_text(docs))
        st.success("PDF is ready for Q&A.")

        for entry in st.session_state.chat_history:
            st.chat_message("user").write(entry["user"])
            st.chat_message("assistant").write(entry["assistant"])

        if st.session_state.pending_question:
            q = st.session_state.pending_question
            st.chat_message("user").write(q)
            with st.spinner("Answering..."):
                ans = answer_question(q, retrieve_docs(q))
            st.chat_message("assistant").write(ans)
            st.session_state.chat_history.append({"user": q, "assistant": ans})
            st.session_state.pending_question = None

        new_q = st.chat_input("Ask a question about the uploaded PDF:", key="chat_input_chat")
        if new_q:
            st.session_state.pending_question = new_q
            st.rerun()
