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

# ---------------------- 全局配置 ----------------------
st.set_page_config(layout="wide")
st.title("📚 Document Research Assistance System")

HF_CACHE = "/root/autodl-tmp/colqwen2-base"
PDF_DIR = "chat-with-pdf/pdfs/"
IMG_DIR = "chat-with-pdf/images/"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

client = OpenAI(
    api_key="f9e888a6-ae48-4ff8-80ba-16165c683753",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# ---------------------- Session State 初始化 ----------------------
for key, default in {
    'uploaded_files_query': None,  # 多文件上传组件返回值
    'file_path_query': None,  # 多文件保存路径列表
    'pdf_submitted_query': False,  # Query 模块提交标志
    'review_button_query': False,  # Query 模块是否触发预览
    'query_processing_done': False,  # 是否已完成图像提取
    'query': None,  # 当前输入的 Query
    'all_images': [],  # 提取的图像列表

    'uploaded_file_chat': None,  # 单文件上传组件返回值
    'file_path_chat': None,  # 单文件保存路径
    'pdf_submitted_chat': False,  # Chat 模块提交标志
    'review_button_chat': False,  # Chat 模块是否触发预览
    'chat_history': [],  # 聊天历史记录
    'pending_question': None  # 等待回答的问题
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------- PDF 上传函数 ----------------------
PDF_DIR = "chat-with-pdf/pdfs/"
os.makedirs(PDF_DIR, exist_ok=True)


def upload_single_pdf(file):
    try:
        path = os.path.join(PDF_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        return path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def upload_multiple_pdfs(files):
    try:
        paths = []
        for file in files:
            path = os.path.join(PDF_DIR, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            paths.append(path)
        return paths
    except Exception as e:
        st.error(f"Error saving files: {e}")
        return []


# ---------------------- PDF Viewer ----------------------
def show_pdf_modal(files, modal_key):
    modal = Modal(key=modal_key, title="📑 Review Uploaded PDF", max_width=850)
    with modal.container():
        for file in files:
            st.write(Path(file).name)
            pdf_viewer(input=file, width=750)


# ---------------------- PDF 转图像 ----------------------
def extract_images_from_pdfs(file_paths):
    all_images = []
    for file_path in file_paths:
        folder = os.path.join(IMG_DIR, Path(file_path).stem)
        os.makedirs(folder, exist_ok=True)
        if not os.listdir(folder):  # 避免重复转换
            pages = convert_from_path(file_path)
            for i, page in enumerate(pages):
                img_path = os.path.join(folder, f"page_{i + 1}.png")
                page.save(img_path, "PNG")
        for img_file in sorted(Path(folder).glob("*.png")):
            all_images.append(Image.open(img_file).convert("RGB"))
    return all_images


# ---------------------- ColQwen2 模型加载 ----------------------
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


# ---------------------- Query 检索 ----------------------
def handle_query(query):
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


# ---------------------- Chat LLM 处理 ----------------------
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


# ---------------------- Chat 与文档 ----------------------
def load_pdf(path):
    return PDFPlumberLoader(path).load()


def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


pdf_text = ""


def index_docs(docs):
    global pdf_text
    pdf_text = "\n\n".join([d.page_content for d in docs])


def retrieve_docs(_): return pdf_text


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


# ---------------------- Layout：左右列 ----------------------
mainCol1, mainCol2 = st.columns(2)

with mainCol1:
    st.header("🔍 PDF Query")

    # 初始化状态
    if 'uploaded_files_query' not in st.session_state:
        st.session_state['uploaded_files_query'] = []

    # Query 上传组件（多文件）
    uploaded_files_query = st.file_uploader(
        "Upload one or more PDFs", type="pdf", accept_multiple_files=True, key="uploader_query"
    )
    if uploaded_files_query:
        st.session_state.uploaded_files_query = uploaded_files_query
        st.session_state.file_path_query = upload_multiple_pdfs(uploaded_files_query)

    if st.session_state.file_path_query:
        if st.button("Review Uploaded PDFs", key="btn_review_query"):
            st.session_state.review_button_query = True

    if st.session_state.review_button_query:
        show_pdf_modal(st.session_state.file_path_query, "popup_query")
        st.session_state.review_button_query = False

    if st.button("Submit Query PDF", key="btn_submit_query") and st.session_state.file_path_query:
        st.session_state.pdf_submitted_query = True
        st.rerun()

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

        # 没有新 query 时，展示上一次结果（防止 Chat 模块刷新后丢失）
        elif st.session_state.get("query_results"):
            st.write(f"💬 You asked: _{st.session_state.query}_")
            for img, score in st.session_state.query_results:
                st.image(img, caption=f"Similarity: {score:.2f}")

with mainCol2:
    st.header("💬 Chat with PDF")

    if 'uploaded_file_chat' not in st.session_state:
        st.session_state['uploaded_file_chat'] = None

    uploaded_file_chat = st.file_uploader(
        "Upload a single PDF", type="pdf", accept_multiple_files=False, key="uploader_chat"
    )
    if uploaded_file_chat:
        st.session_state.uploaded_file_chat = uploaded_file_chat
        st.session_state.file_path_chat = upload_single_pdf(uploaded_file_chat)

    if st.session_state.file_path_chat:
        if st.button("Review Uploaded PDF", key="btn_review_chat"):
            st.session_state.review_button_chat = True

    if st.session_state.review_button_chat:
        show_pdf_modal([st.session_state.file_path_chat], "popup_chat")
        st.session_state.review_button_chat = False

    if st.button("Submit Chat PDF", key="btn_submit_chat") and st.session_state.file_path_chat:
        st.session_state.pdf_submitted_chat = True
        st.rerun()

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