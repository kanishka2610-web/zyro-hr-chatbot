import streamlit as st
import os
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# 1. Page Config
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢", layout="centered")
st.title("🏢 Zyro Dynamics HR Help Desk")
st.markdown("Welcome to the internal HR assistant portal. Ask any policy or handbook related questions below.")

# 2. Pipeline Resource Caching
@st.cache_resource
def init_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs_id_list = list(vectorstore.docstore._dict.values())
    bm25_corpus = [doc.page_content.lower().split(" ") for doc in docs_id_list]
    bm25 = BM25Okapi(bm25_corpus)
    
    rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=512)
    
    return vectorstore, docs_id_list, bm25, rerank_model, llm

try:
    vectorstore, chunks, bm25, rerank_model, llm = init_pipeline()
except Exception as e:
    st.error(f"Initialization Error: Ensure GROQ_API_KEY is set and 'faiss_index' folder is uploaded. {e}")
    st.stop()

# 3. Hybrid Reranker Retrieval
def hybrid_rerank_retrieve(question, top_k_initial=10, top_n_final=3):
    vector_docs = vectorstore.similarity_search(question, k=top_k_initial)
    tokenized_query = question.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_initial]
    bm25_docs = [chunks[idx] for idx in top_bm25_indices]
    
    seen_contents = set()
    candidate_docs = []
    for doc in (vector_docs + bm25_docs):
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            candidate_docs.append(doc)
            
    pairs = [[question, doc.page_content] for doc in candidate_docs]
    rerank_scores = rerank_model.predict(pairs)
    ranked_indices = np.argsort(rerank_scores)[::-1]
    return [candidate_docs[idx] for idx in ranked_indices[:top_n_final]]

# 4. Settings & Prompts
REFUSAL_MESSAGE = "I am sorry, but I can only answer questions related to Zyro Dynamics (Acrux Dynamics) internal HR policies, handbook, leave policies, and work-from-home guidelines. The requested information is outside the scope of my knowledge base."

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert HR assistant for Zyro Dynamics (also referred to as Acrux Dynamics). "
               "Use ONLY the provided context to answer. If the answer is not in the context, "
               "state exactly: 'I am sorry, but I cannot find that information in the company policy documents.' "
               "Be absolute, literal, and precise with metrics, numbers, dates, and names."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# 5. Interface Messaging State
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    q_norm = user_query.strip().lower()
    oos_keywords = ["zoho", "freshworks", "salesforce", "acruxcrm", "revenue last year", "financial performance", "apply for a job", "recruitment and hiring", "careers"]
    
    with st.chat_message("assistant"):
        with st.spinner("Searching company documents..."):
            if any(kw in q_norm for kw in oos_keywords):
                response = REFUSAL_MESSAGE
            else:
                try:
                    retrieved_docs = hybrid_rerank_retrieve(user_query)
                    context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    chain = (
                        {"context": lambda x: context_str, "question": RunnablePassthrough()}
                        | RAG_PROMPT
                        | llm
                        | StrOutputParser()
                    )
                    response = chain.invoke(user_query)
                except Exception as e:
                    response = f"An execution error occurred: {str(e)}"
                    
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
