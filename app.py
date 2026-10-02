import hashlib
import os
import tempfile
import pandas as pd
import streamlit as st

# LangChain Imports (modern modular setup)
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# LangChain 0.3 chain factories
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# --- A. CONFIGURATION ---
MODEL_NAME = "phi3:mini"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# --- B. CORE FUNCTIONS ---

@st.cache_resource
def get_ollama_llm(model_name):
    """Initializes and returns the Ollama LLM."""
    try:
        llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            num_gpu=0,
        )
        return llm
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}")
        return None


def setup_vector_store(uploaded_file):
    """Loads uploaded data, chunks it, and sets up the ChromaDB vector store."""
    if uploaded_file is None:
        return None

    # Save uploaded file temporarily for CSVLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        content = uploaded_file.getvalue()
        if uploaded_file.type == "text/plain":
            csv_content = pd.DataFrame({'Content': [content.decode('utf-8')]}).to_csv(index=False)
            tmp_file.write(csv_content.encode('utf-8'))
        else:
            tmp_file.write(content)
        temp_file_path = tmp_file.name

    try:
        loader = CSVLoader(file_path=temp_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # Using Sentence Transformers for lightweight fast embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=tempfile.mkdtemp(prefix="study-guide-chroma-"),
        )
        
        st.session_state.data_loaded = True
        return vectorstore
    except Exception as e:
        st.error(f"Error processing file or creating vector store: {e}")
        st.exception(e)
        st.session_state.data_loaded = False
        return None
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def get_rag_chain(llm, vectorstore, output_type):
    """Creates modern LCEL retrieval chain based on selected output type."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    if output_type == "Summary":
        template = """You are an educational assistant. Summarize the provided context for a student. 
Focus on key concepts and relationships.

Context:
{context}

Question: Summarize the content.
Summary:"""
    elif output_type == "Q&A":
        template = """You are an educational assistant. Based on the provided context, generate a detailed Q&A set (at least 3 pairs) 
suitable for a study guide.

Context:
{context}

Question: Generate Q&A study pairs.
Q&A Set:"""
    elif output_type == "Flashcards":
        template = """You are an educational assistant. Based on the provided context, generate five flashcards. 
Strictly follow this format:
Front: [Concept]
Back: [Definition]

Context:
{context}

Question: Generate flashcards.
Flashcards:"""
    else:
        template = """You are an expert educational assistant. Use the provided context to answer the user's question. 

Context:
{context}

Question: {input}
Final Answer:"""

    prompt = PromptTemplate.from_template(template)

    # Modern LCEL Chain construction
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return rag_chain


# --- C. STREAMLIT APP ---

def main():
    st.set_page_config(page_title="Local RAG Educational Content Generator", layout="wide")
    st.title("👨‍🏫 Study Content Generator")
    st.caption(f"Powered by Ollama ({MODEL_NAME}) and LangChain")

    llm = get_ollama_llm(MODEL_NAME)
    if llm is None:
        st.stop()

    # Sidebar: File Upload
    st.sidebar.header("1. Upload Lecture Content")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV or TXT file with educational content.",
        type=["csv", "txt"]
    )

    if uploaded_file:
        file_fingerprint = hashlib.sha256(
            uploaded_file.name.encode("utf-8")
            + b"\0"
            + (uploaded_file.type or "").encode("utf-8")
            + b"\0"
            + uploaded_file.getvalue()
        ).hexdigest()

        if st.session_state.get("file_fingerprint") != file_fingerprint:
            st.session_state.pop("vectorstore", None)
            st.session_state.pop("file_fingerprint", None)
            st.session_state.data_loaded = False

            with st.spinner("Processing file and creating vector store..."):
                vectorstore = setup_vector_store(uploaded_file)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.file_fingerprint = file_fingerprint
                    st.sidebar.success("Content processed! Vector Store Ready.")
    else:
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("file_fingerprint", None)
        st.session_state.data_loaded = False

    # Main Interface
    st.header("Generate Educational Materials")
    
    output_type = st.selectbox(
        "Select the type of study material to generate:",
        ["Summary", "Q&A", "Flashcards", "Combined (Full Text Prompt)"],
        index=0
    )

    if output_type == "Combined (Full Text Prompt)":
        user_query = st.text_area("Enter your custom request:", height=100)
        run_button = st.button("Generate Custom Content")
    else:
        user_query = ""
        run_button = st.button(f"Generate {output_type}")

    # Execution Logic
    if run_button and st.session_state.get('vectorstore'):
        with st.spinner(f"Generating {output_type} using {MODEL_NAME}..."):
            rag_chain = get_rag_chain(llm, st.session_state.vectorstore, output_type)
            
            if output_type == "Combined (Full Text Prompt)":
                final_input = user_query if user_query else "Summarize key insights."
            else:
                final_input = f"Generate {output_type} based on the provided content."

            try:
                result = rag_chain.invoke({"input": final_input})
                
                st.subheader(f"✅ Generated {output_type}")
                st.markdown("---")
                st.markdown(result['answer']) 
                st.markdown("---")

            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
                st.exception(e)
    elif run_button and not st.session_state.get('vectorstore'):
        st.warning("Please upload and process a file first in the sidebar (Step 1).")

if __name__ == "__main__":
    main()
