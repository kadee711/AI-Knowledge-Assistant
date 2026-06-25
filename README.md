# 🧠 AI Knowledge Assistant

A Vector-Based Question Answering System built using **Sentence Transformers**, **FAISS**, and **Streamlit**. This project enables users to ask AI-related questions and receive the most relevant answers through semantic similarity search.

---

## 📌 Project Overview

AI Knowledge Assistant is an intelligent chatbot that retrieves answers from a predefined knowledge base using vector embeddings and semantic search. Unlike traditional keyword-based search systems, it understands the meaning of user queries and returns the most relevant response.

The system leverages modern Natural Language Processing (NLP) techniques to transform questions into vector representations and uses FAISS for efficient similarity matching.

---

## 🚀 Features

* Interactive chatbot interface using Streamlit
* Semantic question matching using embeddings
* Fast similarity search with FAISS
* AI-focused knowledge base
* Real-time response generation
* User-friendly web interface
* Lightweight and easy to deploy

---

## 🛠️ Technology Stack

### Frontend

* Streamlit

### Backend

* Python

### Libraries Used

* Sentence Transformers
* FAISS
* Pandas
* NumPy

### NLP Model

* all-MiniLM-L6-v2

---

## 📂 Project Structure

```text
AI-Knowledge-Assistant/
│
├── app.py              # Streamlit Web Application
├── chatbot.py          # Console-Based Chatbot
├── dataset.csv         # Knowledge Base Dataset
├── requirements.txt    # Project Dependencies
└── README.md
```

---

## ⚙️ How It Works

1. The dataset containing questions and answers is loaded.
2. Questions are converted into vector embeddings using Sentence Transformers.
3. Embeddings are indexed using FAISS.
4. User queries are converted into embeddings.
5. FAISS finds the most similar question in the dataset.
6. The corresponding answer is returned to the user.

---

## 📚 Knowledge Base Topics

The chatbot can answer questions related to:

* Artificial Intelligence (AI)
* Machine Learning
* Deep Learning
* Natural Language Processing (NLP)
* Large Language Models (LLMs)
* Retrieval-Augmented Generation (RAG)
* LangChain
* FAISS
* Vector Databases
* Embeddings
* Semantic Search
* Neural Networks
* Chatbots
* Python
* Pandas
* NumPy
* APIs
* Git
* GitHub

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/AI-Knowledge-Assistant.git
cd AI-Knowledge-Assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Open the URL displayed in the terminal (usually http://localhost:8501).

---

## 📸 Sample Questions

* What is Artificial Intelligence?
* What is Machine Learning?
* What is FAISS?
* What is an embedding?
* What is semantic search?
* How does a retrieval-based chatbot work?
* What is LangChain?
* What is GitHub?

---

## 🎯 Learning Outcomes

This project demonstrates:

* Natural Language Processing fundamentals
* Vector embeddings
* Semantic search techniques
* Information retrieval systems
* Streamlit web development
* FAISS indexing and similarity search

---

## 🔮 Future Enhancements

* Multi-document knowledge base
* PDF question answering
* Voice-enabled chatbot
* Generative AI integration
* RAG-based architecture
* Cloud deployment
* Conversation memory

---

