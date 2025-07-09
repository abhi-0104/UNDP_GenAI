# UNDP_GenAI

# 🌍 Climate Change & Financing for Development (FFD) — GenAI Intern Assignment

This repository contains the solution to the SDG AI Lab's GenAI Intern assessment (June 2025), focused on analyzing climate-related policy documents from the Fourth International Conference on Financing for Development (FFD4). The project leverages modern LLM pipelines to extract, explore, and query climate finance data.

---

## 🧠 Project Objectives

- Scrape and parse official FFD4 documents related to climate change and development finance.
- Preprocess and clean the text for LLM analysis.
- Perform Exploratory Data Analysis (EDA) on climate finance themes.
- Query the corpus using Groq's LLaMA-3 LLM for deep insight into policy trends.
- Optionally evaluate LLM outputs based on accuracy, relevance, and clarity.

---

## 🛠️ Tech Stack

- **Python** (3.10+)
- **BeautifulSoup** & **requests** (Scraping)
- **PyMuPDF** (PDF Parsing)
- **spaCy** & **TextBlob** (NER & Sentiment Analysis)
- **FAISS** (Indexing for document retrieval)
- **LangChain** & **Groq API** (LLM integration with LLaMA)
- **Docker** (Containerized deployment)

---

## 🚀 Setup Instructions

###  1. Clone the Repository

```bash
git clone https://github.com/your-username/UNDP_GenAI.git
cd UNDP_GenAI
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a .env file:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

🐳 Optional: Docker Usage
Build and run the Docker container:

```bash
docker build -t undp-genai .
docker run --rm -it --env-file .env -v $(pwd)/faiss_index:/app/faiss_index undp-genai
```

