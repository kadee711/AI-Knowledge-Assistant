import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load CSV
data = pd.read_csv("dataset.csv")

print("CSV Columns:", data.columns)

questions = data[data.columns[0]].tolist()

# Generate embeddings
question_embeddings = model.encode(questions)

# Create FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(question_embeddings))

print("AI FAQ Chatbot Ready!")
print("Type 'exit' to quit.\n")

while True:
    user_query = input("You: ")

    if user_query.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Convert query to embedding
    query_embedding = model.encode([user_query])

    # Search similar questions
    distances, indices = index.search(
        np.array(query_embedding),
        k=1
    )

    best_match = indices[0][0]

    print("Chatbot:", data.iloc[best_match][data.columns[1]])
    print()