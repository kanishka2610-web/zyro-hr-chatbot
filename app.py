import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page Setup
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🤖")
st.title("🤖 Zyro Dynamics HR Help Desk")

# Load Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

# Initialize Components (Cached for performance)
@st.cache_resource
def load_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Note: Ensure your FAISS index is saved in the repo or use the logic from your notebook
    # This example assumes you have a saved 'faiss_index' folder in your repo
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    return retriever, llm

retriever, llm = load_rag_components()

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # RAG Logic
    context = retriever.invoke(question)
    formatted_context = "\n\n".join([doc.page_content for doc in context])
    
    prompt = f"""You are an HR assistant. Answer using this context: {formatted_context}
    Question: {question}"""
    
    answer = llm.invoke(prompt).content

    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
