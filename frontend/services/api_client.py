import requests
from utils.config import BACKEND_URL


def health_check():

    response = requests.get(
        f"{BACKEND_URL}/health"
    )

    return response.json()


def upload_pdfs(files):

    payload = []

    for file in files:

        payload.append(
            (
                "files",
                (
                    file.name,
                    file,
                    "application/pdf",
                ),
            )
        )

    response = requests.post(
        f"{BACKEND_URL}/upload_and_process_pdfs",
        files=payload,
        timeout=600,
    )

    return response.json()


def ask_question(
    question: str,
    top_k: int = 5,
    selected_docs: list = None,
):

    response = requests.post(
        f"{BACKEND_URL}/chat",
        json={
            "message": question,
            "top_k": top_k,
            "documents": selected_docs,
        },
        timeout=300,
    )

    return response.json()


def get_vector_count():

    response = requests.get(
        f"{BACKEND_URL}/vector_store/count"
    )

    return response.json()


def search_vectors(
    query: str,
    top_k: int = 5,
):

    response = requests.post(
        f"{BACKEND_URL}/vector_store/search",
        json={
            "query": query,
            "top_k": top_k,
        },
        timeout=120,
    )

    return response.json()


def fetch_stored_documents() -> list:
    """Fetches the list of all uploaded documents from the backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/documents")
        if response.status_code == 200:
            res_json = response.json()
            if res_json.get("status") == "success":
                return res_json.get("data", [])
        return []
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return []
    
    
def delete_document_from_backend(filename: str) -> bool:
    """Sends a request to delete a document and its vectors from the backend."""
    try:
        response = requests.delete(f"{BACKEND_URL}/documents/{filename}")
        if response.status_code == 200:
            res_json = response.json()
            return res_json.get("status") == "success"
        return False
    except Exception as e:
        print(f"Error deleting document {filename}: {e}")
        return False