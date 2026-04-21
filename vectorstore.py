from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def create_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("Vector store saved.")
    return vectorstore

def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore
from loader import load_document
from chunker import split_documents

docs = load_document("book robo.docx")
chunks = split_documents(docs)

vectorstore = create_vectorstore(chunks)

query = "what is this document about?"
results = vectorstore.similarity_search(query, k=3)

print(f"\nTop 3 most relevant chunks for: '{query}'")
print("---")
for i, result in enumerate(results):
    print(f"Chunk {i+1}:")
    print(result.page_content[:200])
    print("---")