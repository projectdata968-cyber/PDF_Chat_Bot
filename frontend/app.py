import streamlit as st
from collections import defaultdict

from services.api_client import (
    upload_pdfs,
    ask_question,
    get_vector_count,
    health_check,
    fetch_stored_documents, 
    delete_document_from_backend,
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="OCR RAG PDF Assistant",
    page_icon="📄",
    layout="wide",
)

# =========================
# SESSION STATE
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

MAX_HISTORY = 20

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# =========================
# HEADER
# =========================
st.title("📄 OCR + RAG PDF Assistant")
st.markdown(
    """
    Upload PDFs, process them with OCR,
    and ask questions using RAG.
    """
)

# =========================
# BACKEND STATUS
# =========================
try:
    health = health_check()
    if health["status"] == "success":
        st.success("Backend Connected")
    else:
        st.error("Backend Error")
except Exception:
    st.error("Cannot connect to FastAPI backend.")
    st.stop()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("System Status")
    try:
        count_response = get_vector_count()
        if count_response["status"] == "success":
            st.metric(
                "Vector Count",
                count_response["data"],
            )
    except Exception:
        st.warning("Vector store unavailable")

    st.subheader("Stored Documents")
    documents = fetch_stored_documents()

    if not documents:
        st.sidebar.info("No documents uploaded yet.")
    else:
        for doc in documents:
            filename = doc["filename"]
            chunks = doc.get("chunks", 0)
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 **{filename}**")
                st.caption(f"Chunks: {chunks}")
            with col2:
                if st.button("🗑️", key=f"del_{filename}"):
                    if delete_document_from_backend(filename):
                        st.toast(f"Deleted {filename} successfully!")
                        st.rerun()
                    else:
                        st.sidebar.error(f"Failed to delete {filename}")

    # --------------------------
    # Clear Chat Button
    # --------------------------
    st.divider()

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):
        st.session_state.chat_history = []
        st.rerun()

# =========================
# PDF UPLOAD SECTION
# =========================
st.divider()
st.header("Upload PDFs")

uploaded_files = st.file_uploader(
    "Select PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
)

if st.button("Process PDFs", use_container_width=True):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Processing PDFs..."):
            result = upload_pdfs(uploaded_files)

        if result["status"] == "success":
            st.session_state.documents_processed = True
            st.session_state.uploader_key += 1
            st.toast("PDFs processed successfully!")
            st.rerun()
        else:
            st.error(result["message"])


# =========================
# DOCUMENT SELECTION (UNIFIED RAG FILTER)
# =========================

documents = fetch_stored_documents()

available_docs = [
    doc["filename"]
    for doc in documents
]

# None-safe fallback (IMPORTANT)
if available_docs:

    selected_docs = st.multiselect(
        "Select PDFs for search",
        options=available_docs,
        default=available_docs
    )

else:

    selected_docs = None


# =========================
# CHAT SECTION
# =========================
st.divider()
st.header("Chat With PDFs")

top_k = st.slider(
    "Retrieved Chunks",
    min_value=1,
    max_value=20,
    value=5,
)

with st.form("chat_form", clear_on_submit=True):
    question = st.text_input(
        "Ask a question about your PDFs",
        placeholder="Type your question here..."
    )
    submit_button = st.form_submit_button(
        "Ask Question",
        use_container_width=True
    )

if submit_button:
    if not question.strip():
        st.warning("Enter a valid question.")
    else:
        with st.spinner("Generating answer..."):
            response = ask_question(
                question=question,
                top_k=top_k,
                selected_docs=selected_docs,
            )
            # print(response)

        if response["status"] == "success":
            data = response["data"]
            sources = data.get("sources",[])

            st.session_state.chat_history.append(
                {
                    "question": data["question"],
                    "answer": data["answer"],
                    "sources": sources
                }
            )
            if (len(st.session_state.chat_history)> MAX_HISTORY):

                st.session_state.chat_history = (
                    st.session_state.chat_history[
                        -MAX_HISTORY:
                    ]
                )
            st.rerun()

        else:
            st.error(response["message"])

# =========================
# CHAT HISTORY
# =========================
st.divider()
st.header("Conversation")

if not st.session_state.chat_history:

    st.info("No conversation yet.")

else:
    
    for item in reversed(
    st.session_state.chat_history
):

        with st.chat_message("user"):
            st.write(
                item["question"]
            )

        with st.chat_message("assistant"):

            st.markdown(
                item["answer"]
            )

            sources = item.get(
                "sources",
                []
            )

            if sources:

                grouped_sources = (
                    defaultdict(list)
                )

                for source in sources:

                    file_name = source.get(
                        "file_name",
                        "Unknown File"
                    )

                    page = source.get(
                        "page"
                    )

                    if page is not None:
                        grouped_sources[
                            file_name
                        ].append(page)

                st.markdown(
                    "#### 📚 Sources"
                )

                for (
                    file_name,
                    pages
                ) in grouped_sources.items():

                    unique_pages = sorted(
                        set(pages)
                    )

                    page_text = (
                        ", ".join(
                            map(
                                str,
                                unique_pages
                            )
                        )
                    )

                    st.caption(
                        f"📄 {file_name} | Pages: {page_text}"
                    )

            else:

                st.caption(
                    "No source information available"
                )
