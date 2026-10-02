# 👨‍🏫 Study Content Generator

A **Streamlit** app that converts uploaded educational content (CSV/TXT) into interactive study materials using **local Ollama LLM**, **LangChain**, and **ChromaDB**. Ideal for teachers and students to generate personalized study resources offline.

---

## 🚀 Features

- 📝 **Summaries** tailored for high school students  
- ❓ **Q&A Study Guides** with multiple question-answer pairs  
- 💡 **Flashcards** for easy memorization  
- 🛠️ **Custom Queries** with flexible prompts  
- 💾 **Local Deployment** — no cloud API required  
- ⚡ **Fast Retrieval** using ChromaDB vector store  

---

## 🧰 Tech Stack

- Python 🐍  
- Streamlit 🌐  
- LangChain 📚  
- Ollama LLM 🤖  
- Chroma Vector Store 💾  
- SentenceTransformers Embeddings 🔗  

---

## 📝 How to Use

1. Clone the repository:

Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

Upload your CSV or TXT educational content in the sidebar.

Select the type of output: Summary, Q&A, Flashcards, or Custom Query.

Click the button to generate study material!

🤝 Contribution

Contributions are welcome! Feel free to submit issues, suggest features, or make pull requests.

⚠️ Requirements

Local Ollama installation and running ollama serve

Python 3.9+

Compatible CSV/TXT files
