import os
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =============================================================================
# 1. Environment & Page Configuration
# =============================================================================
load_dotenv()

st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🏢",
    layout="centered",
    initial_sidebar_state="expanded"
)

REFUSAL_MESSAGE = (
    "I am a company information assistant and can only answer "
    "questions using the information provided in the company "
    "knowledge base. I don't have sufficient information to answer "
    "that question."
)

# =============================================================================
# 2. Cached RAG Pipeline Loader (Runs once at startup)
# =============================================================================
@st.cache_resource(show_spinner="Initializing Zyro Dynamics HR Knowledge Base & Reranker...")
def load_rag_pipeline():
    # A. Load Embeddings (Must match BAAI/bge-base-en-v1.5 used in Kaggle)
    embeddings_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

    # B. Load Pre-Saved FAISS Vectorstore & Chunks
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # C. Build Hybrid Retriever (75% Semantic MMR + 25% Keyword BM25)
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 30, "lambda_mult": 0.75}
    )
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    base_retriever = EnsembleRetriever(
        retrievers=[mmr_retriever, bm25_retriever],
        weights=[0.75, 0.25]
    )

    # D. Add Cross-Encoder Reranker (Keeps top 4 sharpest chunks)
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=4)

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # E. Initialize Gemini Model
    llm_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_output_tokens=500
    )

    # F. Scope Guardrail Chain
    guardrail_prompt = ChatPromptTemplate.from_template("""
    You are a scope classifier for a company HR assistant.

    IN_SCOPE:
    Questions about company information, HR policies,
    employee benefits, leaves, salary, reimbursement,
    working hours, holidays and workplace rules.

    OUT_OF_SCOPE:
    Questions unrelated to company information,
    including sports, entertainment, general knowledge
    and unrelated programming questions.

    Important:
    A company-related question remains IN_SCOPE even
    if its answer is unavailable in the documents.

    Respond with exactly one word:
    IN_SCOPE or OUT_OF_SCOPE.

    Question: {question}
    """)
    guardrail_chain = guardrail_prompt | llm_model | StrOutputParser()

    # G. Grounded RAG Generation Chain
    rag_prompt = ChatPromptTemplate.from_template("""
    You are a company HR knowledge-base assistant for Zyro Dynamics Pvt. Ltd.
    Answer the user's question using ONLY the information explicitly available in the context.

    RULES:
    1. Never use outside knowledge or assumptions.
    2. Write in clean, plain-text complete sentences without filler phrases like "Based on the context".
    3. Preserve all exact numbers, currency limits (Rs. / USD), dates, percentages, and eligibility conditions verbatim.
    4. If the user asks about "my grade", "my role", or "my eligibility" without specifying which one, provide the entitlements across all grades/categories listed in the context instead of refusing.
    5. Ensure the context genuinely answers the user's question—do not pull unrelated table rows just because they share a keyword.
    6. Only if the context genuinely does not contain the answer, respond with this exact message:

    {refusal_message}

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)
    rag_chain = rag_prompt | llm_model | StrOutputParser()

    return retriever, guardrail_chain, rag_chain


# =============================================================================
# 3. Helper Functions
# =============================================================================
def format_docs(docs) -> str:
    formatted = []
    for d in docs:
        source = os.path.basename(d.metadata.get("source", "Unknown"))
        page = d.metadata.get("page", 0) + 1
        formatted.append(f"[Source: {source} | Page: {page}]\n{d.page_content}")
    return "\n\n--- DOCUMENT CHUNK ---\n\n".join(formatted)


def ask_hr_assistant(question: str, retriever, guardrail_chain, rag_chain) -> dict:
    # Step 1: Check Scope Guardrail
    verdict = guardrail_chain.invoke({"question": question}).strip().upper()
    if "OUT_OF_SCOPE" in verdict:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    # Step 2: Retrieve & Rerank Relevant Chunks
    docs = retriever.invoke(question)
    if not docs:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    # Step 3: Generate Grounded Answer
    context = format_docs(docs)
    answer = rag_chain.invoke({
        "context": context,
        "question": question,
        "refusal_message": REFUSAL_MESSAGE
    }).strip()

    # Do not attach citations if the assistant refused to answer
    if answer == REFUSAL_MESSAGE:
        return {"answer": answer, "sources": []}

    # Deduplicate document + page citations
    seen = set()
    sources = []
    for d in docs:
        file_name = os.path.basename(d.metadata.get("source", "Policy Document"))
        page_num = d.metadata.get("page", 0) + 1
        if (file_name, page_num) not in seen:
            seen.add((file_name, page_num))
            snippet = d.page_content[:240].replace("\n", " ").strip() + "..."
            sources.append({
                "file": file_name,
                "page": page_num,
                "snippet": snippet
            })

    return {"answer": answer, "sources": sources}


# =============================================================================
# 4. Sidebar UI
# =============================================================================
with st.sidebar:
    st.header("🏢 Zyro Dynamics HR")
    st.markdown(
        "Your internal assistant for HR policies, leave rules, "
        "WFH eligibility, travel allowances, and employee benefits."
    )
    st.divider()

    st.subheader("💡 Try a Sample Question")
    sample_questions = [
        "How many casual leaves do I get per year?",
        "Am I eligible to work from home at my grade?",
        "What's my hotel and daily allowance for international travel?",
        "What is the internet reimbursement limit?",
        "How do I apply for a job opening at Zyro Dynamics?",
        "Who won the latest cricket match?"
    ]

    selected_question = None
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            selected_question = q

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =============================================================================
# 5. Main Chat Interface
# =============================================================================
st.title("Zyro Dynamics HR Knowledge Assistant")
st.caption("Grounded Policy Q&A • Hybrid Search (FAISS MMR + BM25) • BGE Reranker")

# Pre-flight validation checks
if not os.getenv("GOOGLE_API_KEY"):
    st.error("⚠️ `GOOGLE_API_KEY` not found! Add `GOOGLE_API_KEY=your_key` to a `.env` file in this folder.")
    st.stop()

if not os.path.exists("faiss_index") or not os.path.exists("chunks.pkl"):
    st.error("⚠️ Missing `faiss_index/` folder or `chunks.pkl` file. Place both in the same directory as `app.py`.")
    st.stop()

# Load pipeline
retriever, guardrail_chain, rag_chain = load_rag_pipeline()

# Initialize session chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to Zyro Dynamics HR Support! Ask me anything about company policies, leave, WFH, payroll, or travel allowances.",
            "sources": []
        }
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 View Policy Sources"):
                for src in message["sources"]:
                    st.markdown(f"- **{src['file']}** (Page {src['page']}): _{src['snippet']}_")

# Handle user input (typed or clicked from sidebar)
prompt = st.chat_input("Ask an HR policy question...") or selected_question

if prompt:
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Render assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            try:
                response = ask_hr_assistant(prompt, retriever, guardrail_chain, rag_chain)
                answer_text = response["answer"]
                source_list = response["sources"]
            except Exception as e:
                answer_text = f"⚠️ An error occurred while processing your request: `{e}`"
                source_list = []

            st.markdown(answer_text)
            if source_list:
                with st.expander("📄 View Policy Sources"):
                    for src in source_list:
                        st.markdown(f"- **{src['file']}** (Page {src['page']}): _{src['snippet']}_")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": source_list
    })