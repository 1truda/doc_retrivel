import os
import base64
import tempfile
import torch
import streamlit as st
from streamlit_modal import Modal
from streamlit_pdf_viewer import pdf_viewer
from pathlib import Path
from PIL import Image
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor
from pdf2image import convert_from_path

from openai import OpenAI

# st.set_page_config(layout="wide")
# LLM = "deepseek-r1:8b"


st.set_page_config(layout="wide")

# ✅ 初始化 session_state 变量，防止 KeyError
if 'pdf_submitted' not in st.session_state:
    st.session_state['pdf_submitted'] = False
if 'file_path' not in st.session_state:
    st.session_state['file_path'] = None
if 'query' not in st.session_state:
    st.session_state['query'] = None
if 'question' not in st.session_state:
    st.session_state['question'] = None

########Sambanova#######
# client = OpenAI(
#     ...

########Sambanova#######
# client = OpenAI(
#     api_key="fee60d77-0151-444e-b008-75efdd59b219",
#     base_url="https://api.sambanova.ai/v1",
# )

########Ark#######
client = OpenAI(
    api_key="f9e888a6-ae48-4ff8-80ba-16165c683753",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


def sambanova_chat(messages):
    try:
        response = client.chat.completions.create(
            # Sambanova
            # model="Meta-Llama-3.1-8B-Instruct",
            # model="DeepSeek-R1-0528",

            # Ark
            model="deepseek-r1-250528",
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[SambaNova API Error] {str(e)}"


# set button style
st.markdown("""
<style>
#root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div:nth-child(4) > div > section > button {width:20%}
#root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5 > div:nth-child(1) > div > div > div > div > div > button {background-color:#17c8d1; width:100%; min-height:50px}
#root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5 > div:nth-child(2) > div > div > div > div > div > button {background-color:#8be05a; width:100%; min-height:50px}        
</style>""", unsafe_allow_html=True)
st.markdown("""
<style>
    .appview-container .main .block-container {{
        padding-top: {padding_top}rem;
        padding-bottom: {padding_bottom}rem;
        padding-left: {padding_left}%;
        padding-right: {padding_right}%;
        }}
</style>""".format(
    padding_top=0, padding_bottom=0, padding_left=12.5, padding_right=12.5
),
    unsafe_allow_html=True,
)

# Prompt template for answering questions
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Directory to save uploaded PDFs
pdfs_directory = "chat-with-pdf/pdfs/"

# Ensure the directory exists
os.makedirs(pdfs_directory, exist_ok=True)

# Initialize embeddings and model
# embeddings = OllamaEmbeddings(model=LLM)
# llm_model = OllamaLLM(model=LLM)

# Initialize vector store
vector_store = None


def click_button():
    st.session_state.button = not st.session_state.button


def upload_pdf(file):
    """Save the uploaded PDF to the specified directory."""
    try:
        file_path = os.path.join(pdfs_directory, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def load_pdf(file_path):
    """Load the content of the PDF using PDFPlumberLoader."""
    try:
        loader = PDFPlumberLoader(file_path)
        return loader.load()
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None


def split_text(documents):
    """Split the documents into smaller chunks for indexing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(documents)


# 全局变量保存 PDF 文本
pdf_full_text = ""


def index_docs(documents):
    """直接将整个 PDF 的文本保存为上下文（不做嵌入）"""
    global pdf_full_text
    pdf_full_text = "\n\n".join([doc.page_content for doc in documents])


def retrieve_docs(query):
    """简化逻辑，返回整个 PDF 文本作为‘上下文’"""
    global pdf_full_text
    return pdf_full_text


def answer_question(question, documents):
    context = documents if isinstance(documents, str) else ""
    MAX_CONTEXT_LENGTH = 12000  # approx 4000 tokens
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "\n...\n[内容已截断]"

    prompt = f"""
You are a professional document question-answering assistant. Please answer the user's question based on the provided document content. Do not make up information. If the answer cannot be found, reply: "The answer is not found in the document."

Document:
{context}

Question:
{question}
"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about PDF documents."},
        {"role": "user", "content": prompt}
    ]
    return sambanova_chat(messages)


# Streamlit UI
st.title("Document Research Assistance System")
uploaded_file = st.file_uploader(
    "Upload a PDF file to start", type="pdf", accept_multiple_files=False
)

if uploaded_file:

    # Save the uploaded PDF
    file_path = upload_pdf(uploaded_file)
    if file_path:
        success1 = st.success(f"File uploaded successfully: {uploaded_file.name}")
        if 'button1' not in st.session_state:
            st.session_state.button1 = False
        modal = Modal(key="Demo Key", title="Review the uploaded PDF", padding=30, max_width=850)
        # After submit, empty this placeholder
        placeholder = st.empty()
        with placeholder:
            btCol1, btCol2, btCol3 = st.columns([3, 3, 8])
            with btCol1:
                open_modal = st.button('Review your PDF', key="btn_review")
            with btCol2:
                submit_pdf = st.button('Submit PDF to System', key="btn_submit")
                if submit_pdf:
                    st.session_state['pdf_submitted'] = True
                    st.session_state['file_path'] = file_path
        if open_modal:
            with modal.container():
                pdf_viewer(input=file_path, width=750)

if st.session_state['pdf_submitted'] and st.session_state['file_path']:
    file_path = st.session_state['file_path']
    st.write("📄 Starting PDF and query processing block...")
    success1.empty()
    placeholder.empty()
    mainCol1, mainCol2, mainCol3 = st.columns([5, 1, 5])
    with mainCol1:
        st.header("PDF Query")
    with mainCol3:
        st.header("Chat with PDF")
    with mainCol1:
        with st.spinner("Processing your PDF..."):
            st.write("🔍 Loading images from folder...")
            # 自动将上传的 PDF 转为图像
            image_folder = os.path.join("chat-with-pdf", "images", Path(file_path).stem)
            os.makedirs(image_folder, exist_ok=True)

            # 如果文件夹中已有图片则跳过转换（防止重复转）
            if len(os.listdir(image_folder)) == 0:
                pages = convert_from_path(file_path)
                for i, page in enumerate(pages):
                    img_path = os.path.join(image_folder, f"page_{i + 1}.png")
                    page.save(img_path, "PNG")

            # 加载图片为 PIL.Image
            images = []
            for filename in sorted(os.listdir(image_folder)):
                if filename.endswith(".png"):
                    img_path = os.path.join(image_folder, filename)
                    image = Image.open(img_path).convert("RGB")
                    images.append(image)

            st.success("PDF processed successfully! Input your queries below.")
        query = st.chat_input("Input a sentence you want to query:")
        if query:
            st.session_state['query'] = query
        query = st.session_state['query']
        for filename in os.listdir(image_folder):
            if filename.endswith(".png"):
                image_path = os.path.join(image_folder, filename)
                image = Image.open(image_path).convert("RGB")
                images.append(image)
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        st.write("🧠 Loading ColQwen2 model and processor...")
        col_model = ColQwen2.from_pretrained(
            "/root/autodl-tmp/colqwen2-base",
            torch_dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        processor = ColQwen2Processor.from_pretrained("/root/autodl-tmp/colqwen2-base")
        st.write("✅ Model loaded.")

        if query:
            st.write(f"💬 Query received: {query}")
            st.write(f"**Your query:** _{query}_")
            with st.spinner("Processing your query..."):
                st.write("📊 Generating embeddings and similarity scoring...")
                queries = [query]
                image_embeddings_list = []
                batch_size = 4

                batch_queries = processor.process_queries([query])
                batch_queries = {k: v.to(col_model.device) for k, v in batch_queries.items()}

                with torch.no_grad():
                    for i in range(0, len(images), batch_size):
                        image_batch = images[i:i + batch_size]
                        batch_images = processor.process_images(image_batch)
                        batch_images = {k: v.to(col_model.device) for k, v in batch_images.items()}
                        outputs = col_model(**batch_images)
                        image_embeddings_list.append(outputs)

                    # 合并所有图像的 embedding
                    image_embeddings = torch.cat(image_embeddings_list, dim=0)
                    query_embeddings = col_model(**batch_queries)
                scores = processor.score_multi_vector(query_embeddings, image_embeddings)
                st.write("✅ Similarity scoring complete.")
                top_idx = scores[0].argmax().item()
                top_image = images[top_idx]
                similarity_score = scores[0][top_idx].item()
                st.image(top_image, caption=f"Most similar image. Similarity: {similarity_score:.2f}")
    with mainCol3:
        st.write("🧾 Starting document chunking and indexing...")
        with st.spinner("Processing your PDF..."):
            documents = load_pdf(file_path)
            if documents:
                chunked_documents = split_text(documents)
                index_docs(chunked_documents)
                st.success("PDF indexed successfully! Ask your questions below.")

        question = st.chat_input("Ask a question about the uploaded PDF:")
        if question:
            st.session_state['question'] = question
        question = st.session_state['question']
        st.write("🗣 Waiting for PDF-related question input...")
        if question:
            st.write(f"💬 PDF question received: {question}")
            st.chat_message("user").write(question)

            with st.spinner("Retrieving relevant information..."):
                related_documents = retrieve_docs(question)
                if related_documents:
                    answer = answer_question(question, related_documents)

                    st.chat_message("assistant").write(answer)
                else:
                    st.chat_message("assistant").write("No relevant information found.")
