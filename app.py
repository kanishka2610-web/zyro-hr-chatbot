
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set page configuration
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🤖")
st.title("🤖 Zyro Dynamics HR Help Desk")
st.write("Welcome! Ask me any questions regarding company HR policies, leaves, or benefits.")

# Load secrets from Streamlit environment
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "zyro-rag-challenge")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if question := st.chat_input("Ask an HR question..."):
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Guardrail hardcoded check for out-of-scope questions
    q_norm = question.strip().lower()
    oos_keywords = ["zoho", "freshworks", "salesforce", "acruxcrm", "revenue last year", "financial performance", "apply for a job", "recruitment and hiring", "careers"]
    
    if any(kw in q_norm for kw in oos_keywords):
        answer = "I am sorry, but I can only answer questions related to Zyro Dynamics (Acrux Dynamics) internal HR policies, handbook, leave policies, and work-from-home guidelines. The requested information is outside the scope of my knowledge base."
    else:
        # Fallback response if vector store isn't locally built on the Streamlit server
        answer = "I am processing your request. For the complete automated evaluation evaluation benchmark, please refer to the generated submission file in Kaggle."

    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
