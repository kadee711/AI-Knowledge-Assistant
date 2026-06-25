import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

data = load_data()

# Make sure column names are correct
questions = data["Question"].astype(str).tolist()

# ---------------------------
# CREATE FAISS INDEX
# ---------------------------
@st.cache_resource
def create_index():
    embeddings = model.encode(questions)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index

index = create_index()

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("📚 Project Information")

    st.info("""
    AI Knowledge Assistant

    🔹 Sentence Transformers

    🔹 FAISS Vector Search

    🔹 Semantic Question Matching

    🔹 AI Knowledge Base
    """)

    st.divider()


# ---------------------------
# MAIN PAGE
# ---------------------------
st.title("🧠 AI Knowledge Assistant")

st.caption(
    "Vector-Based Question Answering System using Sentence Transformers and FAISS"
)


st.markdown("""
### 📚 Topics Covered in the Knowledge Base

✅ Artificial Intelligence (AI)

✅ Machine Learning

✅ Deep Learning

✅ Natural Language Processing (NLP)

✅ Large Language Models (LLMs)

✅ Retrieval-Augmented Generation (RAG)

✅ LangChain

✅ Vector Databases

✅ FAISS

✅ Embeddings

✅ Semantic Search

✅ Neural Networks

✅ Chatbots

✅ Python

✅ Pandas & NumPy

✅ APIs

✅ Git & GitHub

💡 Try asking questions like:

• What is Artificial Intelligence?

• What is FAISS?

• What is RAG?

• What is an embedding?

• How does a retrieval-based chatbot work?

• What is GitHub?
""")



# ---------------------------
# CHAT HISTORY
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------------------
# USER INPUT
# ---------------------------
user_query = st.chat_input("Ask your AI question here...")

if user_query:

    # Display user message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.write(user_query)

    # Search
    query_embedding = model.encode([user_query])

    D, I = index.search(
        np.array(query_embedding),
        k=1
    )

    best_match = I[0][0]
    answer = data.iloc[best_match]["Answer"]

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
