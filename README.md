# 🧠 Research-RAG: Chat with Research Papers using Groq + LangChain

An intelligent PDF-based question-answering system powered by **LLaMA 3 (via Groq API)** and **LangChain**. It allows users to upload research papers and ask questions about the content using a **RAG (Retrieval-Augmented Generation)** pipeline. 

This project is containerized with **Docker** and supports both local and cloud deployment.

---

## 📸 Demo

![Demo Screenshot](./demo/demo.png) <!-- Optional: add a screenshot if available -->

---

## 🚀 Features

- 🔍 Ask questions directly from uploaded **research papers (PDFs)**
- 🧠 Uses **Groq's LLaMA 3** for blazing-fast, high-quality answers
- 🔗 Built on **LangChain's ConversationalRetrievalChain**
- 🗂️ Stores document chunks in a **vector store** for semantic search
- 🧪 Secure `.env`-based API key management
- 🐳 Fully **Dockerized** — easily deploy anywhere

---

## 🏗️ Tech Stack

| Tool        | Purpose                                     |
|-------------|---------------------------------------------|
| 🧠 Groq API | LLaMA 3 model (LLM backend)                 |
| 🦜 LangChain| LLM chaining and RAG pipeline               |
| 📄 PyMuPDF / PDFLoader | Research paper parsing            |
| 🧾 FAISS    | Vector similarity search                    |
| 🐳 Docker   | Containerization and easy deployment        |
| 🖥️ Streamlit or FastAPI | User Interface & API serving     |

---

## 📁 Folder Structure

├── .env.example # Example env config (Groq API key)
├── Dockerfile # Docker image configuration
├── requirements.txt # Python dependencies
└── README.md # This file
├── app.py # Main entrypoint (Streamlit or FastAPI)
├── Rag_pipeline.py # RAG chain with Groq integration


---

## ⚙️ Installation (Local)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Research-RAG.git
cd Research-RAG

2. Setup Environment
Create a .env file based on the template:
# .env
GROQ_API_KEY=your_groq_api_key


Install dependencies:
pip install -r requirements.txt

3. Run the App
python app.py
Or, if using Streamlit:

streamlit run app.py

🐳 Run with Docker
1. Build the Docker Image
docker build -t research-rag .
2. Run the Container
docker run -p 8501:8501 --env-file .env research-rag
If using Docker Hub:
docker pull vardan201/rag
docker run -p 8501:8501 --env GROQ_API_KEY=your_groq_api_key vardan201/rag

🧪 Example Questions
"Summarize the introduction section."

"What methodology is proposed in this paper?"

"List the key results."

"Compare the proposed model with BERT."
🔄 Roadmap / TODO
 Integrate Chat Memory using LangChain

 Add support for multiple PDFs

 Improve UI with conversation history

 Deploy on Hugging Face Spaces / Render

📦 Deployment Ideas
Docker + Cloud VM (AWS, GCP, Render)

Hugging Face Spaces (if converted to Gradio)

Streamlit Cloud

