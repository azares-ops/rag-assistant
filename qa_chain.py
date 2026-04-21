from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from vectorstore import load_vectorstore

def create_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(model="llama3.2")

    prompt_template = """
    You are a helpful assistant. Use the following context to answer the question.
    If you don't know the answer from the context, just say you don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

chain = create_qa_chain()

question = "what is this document about?"
result = chain.invoke({"query": question})

print("\nQuestion:", question)
print("\nAnswer:", result["result"])
print("\nSources used:")
for doc in result["source_documents"]:
    print("---")
    print(doc.page_content[:200])