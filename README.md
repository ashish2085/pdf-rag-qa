# 📄 PDF-QA with Streamlit & RAG

This is a Streamlit app that…
- Uploads a PDF  
- Indexes it (OpenAI embeddings → FAISS)  
- Lets you ask questions against your document  

## Setup

```bash
git clone https://github.com/ashish2085/pdf-rag-qa.git
cd pdf-rag-qa
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
