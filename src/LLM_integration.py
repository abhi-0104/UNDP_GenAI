import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# === 1. Load API Key ===
load_dotenv()
groq_api_key = os.getenv("api")

# === 2. Load FAISS Vectorstore ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# === 3. Initialize Groq LLaMA ===
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"
)

# === 4. Prompt Template ===
custom_prompt = PromptTemplate.from_template("""
You are a policy research assistant working on Financing for Development and Climate Change.
Use ONLY the provided context to answer the question. Be specific and accurate.

Context:
{context}

Question:
{question}
""")

# === 5. Build RetrievalQA Chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": custom_prompt}
)

print("✅ Groq LLaMA chain is ready to receive queries.")
