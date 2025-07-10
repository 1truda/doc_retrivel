import os

import requests
import torch
from PIL import Image
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor

from openai import OpenAI

client = OpenAI(
    api_key="fee60d77-0151-444e-b008-75efdd59b219",
    base_url="https://api.sambanova.ai/v1",
)

# response = client.chat.completions.create(
#     model="Meta-Llama-3.1-8B-Instruct",
#     messages=[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Hello"}],
#     temperature=0.1,
#     top_p=0.1
# )
#
# print(response.choices[0].message.content)

st.set_page_config(layout="wide")
LLM = "deepseek-r1:8b"

def sambanova_chat(messages):

    try:
        response = client.chat.completions.create(
            # model="Meta-Llama-3.1-8B-Instruct",
            model="DeepSeek-R1-0528",
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[SambaNova API Error] {str(e)}"

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
llm_model = OllamaLLM(model=LLM)

# Initialize vector store
vector_store = None


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
你是一个专业的文档问答助手。请根据以下提供的文档内容回答用户问题。不要编造信息，如果找不到答案，就说“根据文档无法回答”。

文档内容：
{context}

用户问题：
{question}
"""
    messages = [
        {"role": "system", "content": "你是一个专业的 PDF 文档问答助手。"},
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
        st.success(f"File uploaded successfully: {uploaded_file.name}")

        col1, col2, col3 = st.columns([4,1,4])
        with col1:
            st.header("PDF Query")

            with st.spinner("Processing your PDF..."):
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

                # Initialize model and processor
                col_model = ColQwen2.from_pretrained(
                    "vidore/colqwen2-v1.0",
                    torch_dtype=torch.float16,
                    device_map="mps",
                    local_files_only=True,
                    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
                ).eval()
                processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

                # Folder path containing the PNG images
                image_folder = os.path.expanduser("~/Desktop/images_folder")  # Replace this with your actual folder path

                # Load all PNG images from the folder
                images = []
                for filename in os.listdir(image_folder):
                    if filename.endswith(".png"):
                        image_path = os.path.join(image_folder, filename)
                        image = Image.open(image_path).convert("RGB")
                        images.append(image)

                # Use user input as the actual query

            st.success("PDF processed successfully! Input your queries below.")
            query = st.chat_input("Input a sentence you want to query:")
            if query:
                queries = [query]
                # Process the inputs
                batch_images = processor.process_images(images).to("cuda")
                batch_queries = processor.process_queries([query]).to("cuda")

                # Forward pass to get embeddings
                with torch.no_grad():
                    image_embeddings = col_model(**batch_images)
                    query_embeddings = col_model(**batch_queries)

                # Compute similarity scores between the user query and all images
                scores = processor.score_multi_vector(query_embeddings, image_embeddings)
                top_idx = scores[0].argmax().item()
                top_image = images[top_idx]
                similarity_score = scores[0][top_idx].item()

                # Display the most relevant image
                st.image(top_image, caption=f"Most similar image. Similarity: {similarity_score:.2f}")

        with col3:
            st.header("Chat with PDF")
            # Load and process the PDF
            with st.spinner("Processing your PDF..."):
                documents = load_pdf(file_path)
                if documents:
                    chunked_documents = split_text(documents)
                    index_docs(chunked_documents)
                    st.success("PDF indexed successfully! Ask your questions below.")

            # # Chat input
            question = st.chat_input("Ask a question about the uploaded PDF:")

            if question:
                st.chat_message("user").write(question)

                with st.spinner("Retrieving relevant information..."):
                    related_documents = retrieve_docs(question)
                    if related_documents:
                        answer = answer_question(question, related_documents)
                        st.chat_message("assistant").write(answer)
                    else:
                        st.chat_message("assistant").write("No relevant information found.")
