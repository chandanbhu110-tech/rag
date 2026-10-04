from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# Groq LLM
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=500
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
    ),
    (
        "human",
        """Context:

{context}

Question:

{question}
"""
    )
])

print("RAG system created")
print("Press 0 to exit")

while True:

    query = input("You: ")

    if query == "0":
        break

    # Retrieve relevant documents
    docs = retriever.invoke(query)
    print("Number of retrieved documents:", len(docs))

    # DEBUG: show retrieved documents
    print("\n--- RETRIEVED DOCUMENTS ---")

    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(doc.page_content[:1000])
        print("Metadata:", doc.metadata)

    # Combine retrieved documents
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Create prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate answer
    response = model.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")