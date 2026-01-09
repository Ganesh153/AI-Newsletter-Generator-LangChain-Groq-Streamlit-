# 📰 AI Newsletter Generator

An AI-powered newsletter generator that automatically:
- Searches the web for relevant articles
- Picks the best sources using an LLM
- Extracts and chunks content
- Summarizes information using RAG
- Generates a Tim Ferriss–style newsletter

Built using **LangChain**, **Groq LLMs**, **FAISS**, and **Streamlit**.

---

## ✨ Features

- 🔍 Web search using Google Serper API
- 🧠 Smart URL selection via LLM reasoning
- 📄 Content extraction from articles
- 🧩 Text chunking and vector storage with FAISS
- ✍️ Newsletter generation in "5-Bullet Friday" style
- 🖥️ Simple Streamlit UI

---

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **Groq LLM (openai/gpt-oss-120b)**
- **FAISS**
- **HuggingFace Embeddings**
- **Streamlit**
- **Google Serper API**

---

## 📁 Project Structure
ai-newsletter-generator/
├── app.py # Streamlit application
├── helpers.py # Core logic (search, RAG, summarization)
├── requirements.txt # Dependencies
├── .env.example # Environment variable template
├── .gitignore
└── README.md


## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ai-newsletter-generator.git
cd ai-newsletter-generator

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt

### 4️⃣ Setup Environment Variables

Create a .env file:

GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key

### 5️⃣ Run the App
streamlit run app.py

