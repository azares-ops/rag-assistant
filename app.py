import streamlit as st
import os
import tempfile
from loader import load_document
from chunker import split_documents
from vectorstore import create_vectorstore, load_vectorstore
from qa_chain import create_qa_chain

st.title("RAG Smart Assistant")
st.write("Upload a document and ask questions about it.")

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Documents"):
        with st.spinner("Reading and processing your documents..."):
            all_docs = []

            for uploaded_file in uploaded_files:
                ext = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                docs = load_document(tmp_path)
                all_docs.extend(docs)
                os.unlink(tmp_path)

            chunks = split_documents(all_docs)
            vectorstore = create_vectorstore(chunks)
            st.session_state.vectorstore_ready = True
            st.success(f"Processed {len(uploaded_files)} file(s) into {len(chunks)} chunks!")

if st.session_state.get("vectorstore_ready"):
    st.divider()
    question = st.text_input("Ask a question about your documents")

    if st.button("Get Answer"):
        if question:
            with st.spinner("Thinking..."):
                chain = create_qa_chain()
                result = chain.invoke({"query": question})

                st.subheader("Answer")
                st.write(result["result"])

                st.subheader("Sources used")
                for i, doc in enumerate(result["source_documents"]):
                    with st.expander(f"Source {i+1}"):
                        st.write(doc.page_content)
        else:
            st.warning("Please type a question first.")