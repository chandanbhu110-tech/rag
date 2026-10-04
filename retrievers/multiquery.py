from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Documents
docs = [
    Document(
        page_content="Gradient descent is an optimization algorithm used in machine learning."
    ),
    Document(
        page_content="Gradient descent minimizes the loss function."
    ),
    Document(
        page_content="Gradient descent is an optimization that minimizes the loss function."
    ),
    Document(
        page_content="Neural networks use gradient descent for training."
    ),
    Document(
        page_content="Support Vector Machines are supervised learning algorithms."
    )
]

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

# Normal retriever
retriever = vectorstore.as_retriever()

# ChatGroq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=500
)

# Multi Query Retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

# Query
query = "What is gradient descent?"

docs = multi_query_retriever.invoke(query)

print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)