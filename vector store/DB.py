from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Documents
docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_book"}
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "DataScience_book"}
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "DL_book"}
    ),
]

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Chroma vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

# Similarity search
result = vectorstore.similarity_search(
    "What is used for data analysis?",
    k=2
)

for r in result:
    print(r.page_content)
    print(r.metadata)

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

retrieved_docs = retriever.invoke("Explain deep learning")

print("\n--- Retrieved Documents ---")

for d in retrieved_docs:
    print(d.page_content)

# ChatGroq
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=500
)

# Combine retrieved documents
context = "\n\n".join(
    d.page_content for d in retrieved_docs
)

prompt = f"""
Answer the question using only the following context.

Context:
{context}

Question:
Explain deep learning.

Answer clearly and concisely.
"""

response = model.invoke(prompt)

print("\n--- AI ANSWER ---")
print(response.content)