import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Page configuration
st.set_page_config(
    page_title="PDF QA with RAG",
    page_icon="📄🤖",
    layout="centered",
)
st.title("📄 PDF Question Answering with RAG")

# Function to extract text from PDF
def load_pdf(file) -> str:
    reader = PdfReader(file)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            texts.append(page_text)
    return "\n".join(texts)

# Cache the PDF text extraction
@st.cache_data(show_spinner=False)
def get_pdf_text(file) -> str:
    return load_pdf(file)

# Cache the retriever creation (vector store)
@st.cache_resource(show_spinner=False)
def create_retriever(text: str):
    # Initialize embeddings and splitter
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY", ""))
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n", " "],
    )
    # Split and embed
    chunks = splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 4})

# Cache the RetrievalQA chain
def get_qa_chain(retriever):
    llm = OpenAI(
        temperature=0,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

# Sidebar: OpenAI API key input
with st.sidebar:
    st.header("🔑 API Key")
    api_key = st.text_input(
        "OpenAI API Key", type="password", help="Enter your OpenAI API key."
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# Main app flow
uploaded_file = st.file_uploader(
    label="Upload a PDF", type=["pdf"], help="Select a PDF to extract and index."
)

if uploaded_file:
    st.info("Extracting text and building index. This may take a moment...")
    # Extract text and build retriever
    raw_text = get_pdf_text(uploaded_file)
    retriever = create_retriever(raw_text)
    qa_chain = get_qa_chain(retriever)
    st.success("✅ PDF indexed! You can now ask questions.")

    # Ask questions
    user_question = st.text_input("Ask a question about the PDF:")
    if user_question:
        if st.button("Get Answer"):
            with st.spinner("Generating answer..."):
                answer = qa_chain.run(user_question)
            st.write("**Answer:**")
            st.write(answer)

else:
    st.write("Please upload a PDF to get started.")

# Footer
st.markdown("---")
st.markdown("Built with 💖 using Streamlit and LangChain.")
