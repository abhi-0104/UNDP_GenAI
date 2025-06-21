import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# === 1. Load Documents from extracted_texts/ ===
docs_path = "extracted_texts"
all_docs = []

for filename in os.listdir(docs_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(docs_path, filename)
        loader = TextLoader(file_path, encoding="utf-8")
        all_docs.extend(loader.load())

print(f"Loaded {len(all_docs)} documents.")

# === 2. Split Documents ===
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(all_docs)
print(f"Split into {len(split_docs)} chunks.")

# === 3. Embed with Sentence Transformers ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === 4. Create FAISS Vectorstore ===
vectorstore = FAISS.from_documents(split_docs, embedding_model)

# === 5. Save Vectorstore to disk ===
faiss_dir = "faiss_index"
vectorstore.save_local(faiss_dir)
print(f"✅ FAISS index saved to '{faiss_dir}/'")
