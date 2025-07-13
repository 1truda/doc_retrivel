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

st.set_page_config(layout="wide")
LLM = "deepseek-r1:8b"

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
    padding_top=0, padding_bottom=0, padding_left=10, padding_right=10
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
embeddings = OllamaEmbeddings(model=LLM)
llm_model = OllamaLLM(model=LLM)

# Initialize vector store
vector_store = None


def click_button():
    st.session_state.button = not st.session_state.button


def upload_single_pdf(file):
    """Save the uploaded PDF to the specified directory."""
    try:
        file_path = os.path.join(pdfs_directory, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def upload_multiple_pdf(files):
    """Save the uploaded PDF to the specified directory."""
    try:
        file_path_list = []
        for i in range (0,len(files)):
            file_path_list.append(os.path.join(pdfs_directory, files[i].name))
            with open(file_path_list[i], "wb") as f:
                f.write(files[i].getbuffer())
        return file_path_list
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


def index_docs(documents):
    """Index the documents in the vector store."""
    global vector_store
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)


def retrieve_docs(query):
    """Retrieve relevant documents based on the query."""
    return vector_store.similarity_search(query)


def answer_question(question, documents):
    """Generate an answer to the question using the retrieved documents."""
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = f"""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:"""
    return llm_model.generate([prompt]).generations[0][0].text


if 'pdf_submitted_query' not in st.session_state:
    st.session_state['pdf_submitted_query'] = False
if 'file_path_query' not in st.session_state:
    st.session_state['file_path_query'] = None
if 'review_button_query' not in st.session_state:
    st.session_state['review_button_query'] = False

if 'pdf_submitted_chat' not in st.session_state:
    st.session_state['pdf_submitted_chat'] = False
if 'file_path_chat' not in st.session_state:
    st.session_state['file_path_chat'] = None
if 'review_button_chat' not in st.session_state:
    st.session_state['review_button_chat'] = None

# Streamlit UI

st.title("Document Research Assistance System")
mainCol1, mainCol2 = st.columns([1, 1])
with mainCol1:
    st.header("PDF Query")        
    uploaded_file_query = st.file_uploader(
        "Upload **one or multiple** PDF file to start PDF query", type="pdf", accept_multiple_files=True
    )
with mainCol2:
    st.header("Chat with PDF")
    uploaded_file_chat = st.file_uploader(
        "Upload **a single** PDF file to start PDF chat", type="pdf", accept_multiple_files=False
    )

with mainCol1:
    if uploaded_file_query:
        # Save the uploaded PDF
        file_path_list = upload_multiple_pdf(uploaded_file_query)
        if file_path_list:
            for i in range (0,len(file_path_list)):
                txt=f"File uploaded successfully: {uploaded_file_query[i].name}"
                htmlstr1=f"""<p style='background-color:#ebf9ee; color:#3e8750; font-size:14px; border-radius:4px;
                                                        line-height:32px; padding-left:18px; opacity:1'> {txt}
                                                        </style><br></p>""" 
                st.markdown(htmlstr1,unsafe_allow_html=True) 

            modal = Modal(key="popup_1", title="Review the uploaded PDF", max_width=850)
            btCol1, btCol2, btCol3 = st.columns([3, 3, 2])
            with btCol1:
                open_modal = st.button('Review your PDF', key="btn_review")
                if open_modal:
                # if option:
                    st.session_state['review_button_query'] = True
            with btCol2:
                submit_pdf = st.button('Submit PDF to System', key="btn_submit")
                if submit_pdf:
                    st.session_state['pdf_submitted_query'] = True
                    st.session_state['file_path_query'] = file_path_list

if st.session_state['review_button_query']:
    with modal.container():
        for i in range(0,len(file_path_list)):
            st.write(f"{i+1}. {uploaded_file_query[i].name}")
            pdf_viewer(input=file_path_list[i], width=750)
            st.session_state['review_button_query'] = False

with mainCol2:
    if uploaded_file_chat:
        file_path_single = upload_single_pdf(uploaded_file_chat)
        if file_path_single:
            txt=f"File uploaded successfully: {uploaded_file_chat.name}"
            htmlstr1=f"""<p style='background-color:#ebf9ee; color:#3e8750; font-size:14px; border-radius:4px;
                                                    line-height:32px; padding-left:18px; opacity:1'> {txt}
                                                    </style><br></p>""" 
            st.markdown(htmlstr1,unsafe_allow_html=True) 
            modal = Modal(key="popup_2", title="Review the uploaded PDF", max_width=850)
            btCol4, btCol5, btCol6 = st.columns([3, 3, 2])
            with btCol4:
                open_modal = st.button('Review your PDF', key="btn_review_1")
                if open_modal:
                    with st.modal("My First Modal"):
                        st.write("This is a simple modal!")
                        close_button = st.button("Close Modal")
                        if close_button:
                            st.session_state.modal_open = False
                            #st.session_state['review_button_chat'] = True
            with btCol5:
                submit_pdf = st.button('Submit PDF to System', key="btn_submit_1")
                if submit_pdf:
                    st.session_state['pdf_submitted_chat'] = True
                    st.session_state['file_path_chat'] = file_path_single
if st.session_state['review_button_chat']:
    with modal.container():
        st.write(uploaded_file_chat.name)
        pdf_viewer(input=file_path_single, width=750)
        st.session_state['review_button_chat'] = False

with mainCol1:
    if st.session_state['pdf_submitted_query'] and st.session_state.get('file_path_query', None):
        file_path_list = st.session_state['file_path_query']
        st.write("📄 Starting PDF and query processing block...")

        with st.spinner("Processing your PDFs..."):
            all_images = []  # 用于存储所有 PDF 的图片
            for file_path in file_path_list:
                st.write(f"🔍 Processing file: {file_path}...")
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

                all_images.extend(images)  # 将当前 PDF 的图片添加到总列表中

            st.success("All PDFs processed successfully! Input your queries below.")
    query = st.chat_input("Input a sentence you want to query:")
    if query:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        st.write("🧠 Loading ColQwen2 model and processor...")
        col_model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
        st.write("✅ Model loaded.")
        
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
                for i in range(0, len(all_images), batch_size):
                    image_batch = all_images[i:i + batch_size]
                    batch_images = processor.process_images(image_batch)
                    batch_images = {k: v.to(col_model.device) for k, v in batch_images.items()}
                    outputs = col_model(**batch_images)
                    image_embeddings_list.append(outputs)

                # 合并所有图像的 embedding
                image_embeddings = torch.cat(image_embeddings_list, dim=0)
                query_embeddings = col_model(**batch_queries)

            scores = processor.score_multi_vector(query_embeddings, image_embeddings)
            st.write("✅ Similarity scoring complete.")

            # 获取最高得分
            top_idx = scores[0].argsort(descending=True)[:3]
            highest_score = scores[0][top_idx[0]].item()

            if highest_score < 1.0:
                # 如果最高得分低于10分，显示提示信息
                st.write("❌ This PDF file DOESN’T seem to have the information you're looking for :(")
            else:
                # 显示与查询最相关的图片
                for idx in top_idx:
                    top_image = all_images[idx.item()]
                    similarity_score = scores[0][idx].item()
                    st.image(top_image, caption=f"Most similar image. Similarity: {similarity_score:.2f}")
                    
with mainCol2:
    if st.session_state['pdf_submitted_chat'] and st.session_state['file_path_chat']:
            st.write("🧾 Starting document chunking and indexing...")
            with st.spinner("Processing your PDF..."):
                documents = load_pdf(file_path_single)
                if documents:
                    chunked_documents = split_text(documents)
                    index_docs(chunked_documents)
                    st.success("PDF indexed successfully! Ask your questions below.")

    if st.session_state['pdf_submitted_chat'] and st.session_state['file_path_chat']:
        question = st.chat_input("Ask a question about the uploaded PDF:")
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