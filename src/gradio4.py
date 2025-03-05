#python -m src.gradio

# Gradio Framework for lillisa server 
#python -m src.gradio

# Gradio Framework for lillisa server 
import os
import io
import shutil
import time
import uuid
import zipfile
import gradio as gr
import jwt
import tiktoken
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as OpenAI_Llama
import lancedb
from speedict import Rdict
from src import utils
from src.agent_and_tools import (
    PRODUCT,
    answer_from_document_retrieval,
    handle_user_answer,
    improve_query,
    update_retriever,
    get_matching_versions,
    IDDM_PRODUCT_VERSIONS,
    IDA_PRODUCT_VERSIONS
)
from src.lillisa_server_context import LOCALE, LilLisaServerContext
from src.llama_index_lancedb_vector_store import LanceDBVectorStore
from src.llama_index_markdown_reader import MarkdownReader
import git
import re
import traceback

# --- Global Variables ---
REACT_AGENT_PROMPT = None
LANCEDB_FOLDERPATH = None
AUTHENTICATION_KEY = None
DOCUMENTATION_FOLDERPATH = None
QA_PAIRS_GITHUB_REPO_URL = None
QA_PAIRS_FOLDERPATH = None
DOCUMENTATION_NEW_VERSIONS = None
DOCUMENTATION_EOC_VERSIONS = None
DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS = None
DOCUMENTATION_IA_PRODUCT_VERSIONS = None
DOCUMENTATION_IA_SELFMANAGED_VERSIONS = None

# Load environment variables and initialize globals
lillisa_server_env = utils.LILLISA_SERVER_ENV_DICT

if react_agent_prompt_filepath := lillisa_server_env["REACT_AGENT_PROMPT_FILEPATH"]:
    with open(str(react_agent_prompt_filepath), "r", encoding="utf-8") as file:
        REACT_AGENT_PROMPT = file.read()
else:
    raise ValueError("REACT_AGENT_PROMPT_FILEPATH not found in environment")

if lancedb_folderpath := lillisa_server_env["LANCEDB_FOLDERPATH"]:
    LANCEDB_FOLDERPATH = str(lancedb_folderpath)
else:
    raise ValueError("LANCEDB_FOLDERPATH not found in environment")

if authentication_key := lillisa_server_env["AUTHENTICATION_KEY"]:
    AUTHENTICATION_KEY = str(authentication_key)
else:
    raise ValueError("AUTHENTICATION_KEY not found in environment")

if documentation_folderpath := lillisa_server_env["DOCUMENTATION_FOLDERPATH"]:
    DOCUMENTATION_FOLDERPATH = str(documentation_folderpath)
else:
    raise ValueError("DOCUMENTATION_FOLDERPATH not found in environment")

if qa_pairs_github_repo_url := lillisa_server_env["QA_PAIRS_GITHUB_REPO_URL"]:
    QA_PAIRS_GITHUB_REPO_URL = str(qa_pairs_github_repo_url)
else:
    raise ValueError("QA_PAIRS_GITHUB_REPO_URL not found in environment")

if qa_pairs_folderpath := lillisa_server_env["QA_PAIRS_FOLDERPATH"]:
    QA_PAIRS_FOLDERPATH = str(qa_pairs_folderpath)
else:
    raise ValueError("QA_PAIRS_FOLDERPATH not found in environment")

if documentation_new_versions := lillisa_server_env["DOCUMENTATION_NEW_VERSIONS"]:
    DOCUMENTATION_NEW_VERSIONS = str(documentation_new_versions).split(", ")
else:
    raise ValueError("DOCUMENTATION_NEW_VERSIONS not found in environment")

if documentation_eoc_versions := lillisa_server_env["DOCUMENTATION_EOC_VERSIONS"]:
    DOCUMENTATION_EOC_VERSIONS = str(documentation_eoc_versions).split(", ")
else:
    raise ValueError("DOCUMENTATION_EOC_VERSIONS not found in environment")

if documentation_identity_analytics_versions := lillisa_server_env["DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS"]:
    DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS = str(documentation_identity_analytics_versions).split(", ")
else:
    raise ValueError("DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS not found in environment")

if documentation_ia_product_versions := lillisa_server_env["DOCUMENTATION_IA_PRODUCT_VERSIONS"]:
    DOCUMENTATION_IA_PRODUCT_VERSIONS = str(documentation_ia_product_versions).split(", ")
else:
    raise ValueError("DOCUMENTATION_IA_PRODUCT_VERSIONS not found in environment")

if documentation_ia_selfmanaged_versions := lillisa_server_env["DOCUMENTATION_IA_SELFMANAGED_VERSIONS"]:
    DOCUMENTATION_IA_SELFMANAGED_VERSIONS = str(documentation_ia_selfmanaged_versions).split(", ")
else:
    raise ValueError("DOCUMENTATION_IA_SELFMANAGED_VERSIONS not found in environment")

# --- Enhanced Custom CSS for Gradio Interface ---
custom_css = """
/* Global Styles */
:root {
    --primary: #4F46E5;
    --primary-light: #818CF8;
    --primary-dark: #4338CA;
    --secondary: #10B981;
    --accent: #F472B6;
    --gray-100: #F3F4F6;
    --gray-200: #E5E7EB;
    --gray-300: #D1D5DB;
    --gray-500: #6B7280;
    --gray-700: #374151;
    --gray-900: #111827;
    --success: #059669;
    --warning: #F59E0B;
    --error: #DC2626;
    --info: #2563EB;
    --border-radius: 12px;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', sans-serif;
}

/* Importing Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body {
    background: linear-gradient(135deg, #F5F7FF, #EEF2FF);
    font-family: var(--font-primary);
    margin: 0;
    padding: 0;
    color: var(--gray-900);
    min-height: 100vh;
}

/* App Container */
#app {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
}

/* Tabs Styling */
.tabs {
    margin-top: 2rem;
    background-color: white;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-lg);
    overflow: hidden;
}

.tab-nav {
    display: flex;
    background: var(--gray-100);
    border-bottom: 1px solid var(--gray-200);
}

.tab-button {
    padding: 1rem 1.5rem;
    font-weight: 600;
    color: var(--gray-500);
    border: none;
    background: transparent;
    cursor: pointer;
    transition: all 0.3s ease;
}

.tab-button.active {
    color: var(--primary);
    border-bottom: 3px solid var(--primary);
}

.tab-button:hover:not(.active) {
    color: var(--gray-700);
    background-color: var(--gray-200);
}

/* Header Styling */
.header {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 2rem;
}

.header-content {
    text-align: center;
}

.app-logo {
    height: 80px;
    margin-bottom: 1rem;
}

.app-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.app-subtitle {
    color: var(--gray-500);
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* Chatbot Area */
.chatbot-container {
    background-color: white;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-md);
    padding: 1.5rem;
    max-width: 100%;
    height: 500px;
    overflow-y: auto;
}

.chat-message {
    display: flex;
    margin-bottom: 1rem;
    animation: fadeIn 0.3s ease-in-out;
}

.user-message {
    justify-content: flex-end;
}

.assistant-message {
    justify-content: flex-start;
}

.message-bubble {
    max-width: 80%;
    padding: 1rem;
    border-radius: 18px;
    box-shadow: var(--shadow-sm);
    line-height: 1.5;
}

.user-bubble {
    background-color: var(--primary);
    color: white;
    border-bottom-right-radius: 4px;
}

.assistant-bubble {
    background-color: var(--gray-100);
    color: var(--gray-900);
    border-bottom-left-radius: 4px;
}

/* Input Area */
.input-area {
    display: flex;
    margin-top: 1rem;
    gap: 0.5rem;
}

.input-box {
    flex: 1;
    border: 2px solid var(--gray-300);
    border-radius: var(--border-radius);
    padding: 1rem;
    font-size: 1rem;
    transition: border-color 0.3s ease;
}

.input-box:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
}

/* Buttons */
button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.75rem 1.5rem;
    border-radius: var(--border-radius);
    font-weight: 600;
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 0.025em;
    transition: all 0.2s ease;
    cursor: pointer;
    border: none;
    box-shadow: var(--shadow-sm);
}

button:focus {
    outline: none;
}
.primary-button {
    background-color: var(--primary);
    color: white;
}

.primary-button:hover {
    background-color: var(--primary-dark);
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

.secondary-button {
    background-color: white;
    color: var(--gray-700);
    border: 1px solid var(--gray-300);
}

.secondary-button:hover {
    background-color: var(--gray-100);
    transform: translateY(-1px);
}

.success-button {
    background-color: var(--success);
    color: white;
}

.success-button:hover {
    background-color: #047857;
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

.warning-button {
    background-color: var(--warning);
    color: white;
}

.warning-button:hover {
    background-color: #D97706;
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

/* Dropdowns and Form Controls */
.form-control {
    margin-bottom: 1.5rem;
}

.form-label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
    color: var(--gray-700);
}

.dropdown {
    width: 100%;
    padding: 0.75rem 1rem;
    border: 2px solid var(--gray-300);
    border-radius: var(--border-radius);
    background-color: white;
    font-size: 1rem;
    color: var(--gray-900);
    transition: all 0.3s ease;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%236B7280'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 1rem center;
    background-size: 1.5rem;
    padding-right: 2.5rem;
}

.dropdown:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
}

/* Checkbox and Radio styles */
.checkbox-container, .radio-container {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
}

.checkbox, .radio {
    margin-right: 0.5rem;
    width: 1.25rem;
    height: 1.25rem;
    border: 2px solid var(--gray-300);
    border-radius: var(--border-radius);
    appearance: none;
    background-color: white;
    display: grid;
    place-content: center;
    transition: all 0.2s ease;
}

.checkbox:checked, .radio:checked {
    background-color: var(--primary);
    border-color: var(--primary);
}

.checkbox:checked::before {
    content: "";
    width: 0.65em;
    height: 0.65em;
    transform: scale(1);
    background-color: white;
    clip-path: polygon(14% 44%, 0 65%, 50% 100%, 100% 16%, 80% 0%, 43% 62%);
}
.radio {
    border-radius: 50%;
}

.radio:checked::before {
    content: "";
    width: 0.5em;
    height: 0.5em;
    border-radius: 50%;
    transform: scale(1);
    background-color: white;
}

/* Admin Panel */
.admin-panel {
    background-color: white;
    border-radius: var(--border-radius);
    padding: 2rem;
    box-shadow: var(--shadow-md);
}

.admin-header {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
    color: var(--gray-900);
    border-bottom: 2px solid var(--gray-200);
    padding-bottom: 1rem;
}

.admin-section {
    margin-bottom: 2rem;
}

.admin-form {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}

.admin-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
}

/* Notification and Status */
.status-box {
    padding: 1rem;
    border-radius: var(--border-radius);
    margin-top: 1rem;
    animation: fadeIn 0.3s ease-in-out;
}

.status-success {
    background-color: rgba(5, 150, 105, 0.1);
    border-left: 4px solid var(--success);
    color: var(--success);
}

.status-error {
    background-color: rgba(220, 38, 38, 0.1);
    border-left: 4px solid var(--error);
    color: var(--error);
}

.status-info {
    background-color: rgba(37, 99, 235, 0.1);
    border-left: 4px solid var(--info);
    color: var(--info);
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(79, 70, 229, 0); }
    100% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0); }
}

/* Loading State */
.loading {
    display: inline-block;
    width: 1.5rem;
    height: 1.5rem;
    border: 3px solid rgba(79, 70, 229, 0.3);
    border-radius: 50%;
    border-top-color: var(--primary);
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Responsive Adjustments */
@media (max-width: 768px) {
    .app-title {
        font-size: 2rem;
    }
    
    .admin-form {
        grid-template-columns: 1fr;
    }
    
    .admin-actions {
        flex-direction: column;
    }
    
    button {
        width: 100%;
    }
    
    .chatbot-container {
        height: 400px;
    }
}

/* Custom Elements */
.session-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.75rem;
    background-color: var(--gray-100);
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--gray-700);
    margin-bottom: 1rem;
}

.session-badge.active {
    background-color: rgba(16, 185, 129, 0.1);
    color: var(--success);
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--success);
    margin-right: 0.5rem;
    animation: pulse 2s infinite;
}

/* Tooltip */
.tooltip {
    position: relative;
    display: inline-block;
}

.tooltip .tooltip-text {
    visibility: hidden;
    width: 200px;
    background-color: var(--gray-900);
    color: white;
    text-align: center;
    border-radius: var(--border-radius);
    padding: 0.5rem;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 0.75rem;
}

.tooltip:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}

/* File Upload */
.file-upload {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border: 2px dashed var(--gray-300);
    border-radius: var(--border-radius);
    padding: 2rem 1rem;
    transition: all 0.3s ease;
    cursor: pointer;
}

.file-upload:hover {
    border-color: var(--primary);
    background-color: rgba(79, 70, 229, 0.05);
}

.file-upload-icon {
    font-size: 2rem;
    color: var(--gray-500);
    margin-bottom: 1rem;
}

.file-upload-text {
    color: var(--gray-700);
    font-weight: 500;
}

.file-upload-info {
    font-size: 0.875rem;
    color: var(--gray-500);
    margin-top: 0.5rem;
}
"""

# --- Helper Functions ---
def get_llsc(session_id: str, locale: LOCALE = None, product: PRODUCT = None) -> LilLisaServerContext:
    """Retrieve or create a LilLisaServerContext for the given session."""
    db_folderpath = LilLisaServerContext.get_db_folderpath(session_id)
    try:
        keyvalue_db = Rdict(db_folderpath)
        llsc = keyvalue_db[session_id] if session_id in keyvalue_db else None
    finally:
        keyvalue_db.close()

    if not llsc:
        if not (locale and product):
            raise ValueError("Locale and Product are required to initiate a new conversation.")
        llsc = LilLisaServerContext(session_id, locale, product)
    return llsc

def save_llsc(llsc: LilLisaServerContext, session_id: str):
    """Save the LilLisaServerContext back to the database."""
    db_folderpath = LilLisaServerContext.get_db_folderpath(session_id)
    try:
        keyvalue_db = Rdict(db_folderpath)
        keyvalue_db[session_id] = llsc
    finally:
        keyvalue_db.close()

def generate_unique_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())

# --- Function Definitions (Endpoints) ---

### /invoke/
def invoke(session_id: str, locale: str, product: str, nl_query: str, is_expert_answering: bool) -> str:
    try:
        utils.logger.info("session_id: %s, locale: %s, product: %s, nl_query: %s", session_id, locale, product, nl_query)
        llsc = get_llsc(session_id, LOCALE.get_locale(locale), PRODUCT.get_product(product))

        if is_expert_answering:
            llsc.add_to_conversation_history("Expert", nl_query)
            save_llsc(llsc, session_id)
            return nl_query

        conversation_history_list = llsc.conversation_history
        conversation_history = ""
        for poster, message in conversation_history_list:
            conversation_history += f"{poster}: {message}\n"

        improve_query_tool = FunctionTool.from_defaults(fn=improve_query)
        answer_from_document_retrieval_tool = FunctionTool.from_defaults(fn=answer_from_document_retrieval, return_direct=True)
        handle_user_answer_tool = FunctionTool.from_defaults(fn=handle_user_answer, return_direct=True)

        llm = OpenAI_Llama(model="gpt-4o-mini")
        react_agent = ReActAgent.from_tools(
            tools=[handle_user_answer_tool, improve_query_tool, answer_from_document_retrieval_tool],
            llm=llm,
            verbose=True if utils.LOG_LEVEL == utils.logging.DEBUG else False
        )

        react_agent_prompt = (
            REACT_AGENT_PROMPT.replace("<PRODUCT>", product)
            .replace("<CONVERSATION_HISTORY>", conversation_history)
            .replace("<QUERY>", nl_query)
        )
        response = react_agent.chat(react_agent_prompt).response
        llsc.add_to_conversation_history("User", nl_query)
        llsc.add_to_conversation_history("Assistant", response)
        save_llsc(llsc, session_id)
        return response
    except Exception as exc:
        traceback.print_exc()
        utils.logger.critical("Internal error in invoke(): %s", exc)
        return f"Internal error: {str(exc)}"

### /record_endorsement/
def record_endorsement(session_id: str, is_expert: bool) -> str:
    try:
        utils.logger.info("session_id: %s, is_expert: %s", session_id, is_expert)
        llsc = get_llsc(session_id)
        llsc.record_endorsement(is_expert)
        save_llsc(llsc, session_id)
        return "Conversation successfully endorsed! Thank you for your feedback."
    except Exception as exc:
        traceback.print_exc()
        utils.logger.critical("Internal error in record_endorsement(): %s", exc)
        return f"Internal error: {str(exc)}"

### /get_golden_qa_pairs/
def get_golden_qa_pairs(product: str, encrypted_key: str) -> str:
    try:
        jwt.decode(encrypted_key, AUTHENTICATION_KEY, algorithms="HS256")
        if os.path.exists(QA_PAIRS_FOLDERPATH):
            shutil.rmtree(QA_PAIRS_FOLDERPATH)
        git.Repo.clone_from(QA_PAIRS_GITHUB_REPO_URL, QA_PAIRS_FOLDERPATH)
        filepath = f"{QA_PAIRS_FOLDERPATH}/{product.lower()}_qa_pairs.md"
        if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
            return content
        return "No QA pairs found for this product. Please check the repository or try another product."
    except jwt.exceptions.InvalidSignatureError:
        return "Authentication failed: Invalid signature. Please check your encryption key and try again."
    except Exception as exc:
        traceback.print_exc()
        utils.logger.critical("Internal error in get_golden_qa_pairs(): %s", exc)
        return f"Internal error: {str(exc)}"

### /update_golden_qa_pairs/
def update_golden_qa_pairs(product: str, encrypted_key: str) -> str:
    try:
        jwt.decode(encrypted_key, AUTHENTICATION_KEY, algorithms="HS256")
        if os.path.exists(QA_PAIRS_FOLDERPATH):
            shutil.rmtree(QA_PAIRS_FOLDERPATH)
        git.Repo.clone_from(QA_PAIRS_GITHUB_REPO_URL, QA_PAIRS_FOLDERPATH)
        filepath = f"{QA_PAIRS_FOLDERPATH}/{product.lower()}_qa_pairs.md"
        with open(filepath, "r", encoding="utf-8") as file:
            file_content = file.read()

        db = lancedb.connect(LANCEDB_FOLDERPATH)
        table_name = product + "_QA_PAIRS"
        try:
            db.drop_table(table_name)
        except Exception:
            utils.logger.exception(f"Table {table_name} seems to have been deleted. Continuing with insertion process")

        qa_pairs = file_content.split("# Question/Answer Pair")
        qa_pairs = [pair.strip() for pair in qa_pairs if pair.strip()]

        documents = []
        qa_pattern = re.compile(r"Question:\s*(.*?)\nAnswer:\s*(.*)", re.DOTALL)

        if product == "IDDM":
            product_versions = IDDM_PRODUCT_VERSIONS
            version_pattern = re.compile(r"v?\d+\.\d+", re.IGNORECASE)
        else:
            product_versions = IDA_PRODUCT_VERSIONS
            version_pattern = re.compile(r"\b(?:IAP[- ]\d+\.\d+|version[- ]\d+\.\d+|descartes(?:-dev)?)\b", re.IGNORECASE)

        for pair in qa_pairs:
            match = qa_pattern.search(pair)
            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                document = Document(text=f"{question}")
                document.metadata["answer"] = answer
                matched_versions = get_matching_versions(question, product_versions, version_pattern)
                document.metadata["version"] = matched_versions[0] if matched_versions else "none"
                document.excluded_embed_metadata_keys.append("version")
                document.excluded_embed_metadata_keys.append("answer")
                documents.append(document)

        splitter = SentenceSplitter(chunk_size=10000)
        nodes = splitter.get_nodes_from_documents(documents=documents, show_progress=True)

        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        vector_store = LanceDBVectorStore(uri="lancedb", table_name=table_name, query_type="hybrid")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        retriever = index.as_retriever(similarity_top_k=8)
        update_retriever(table_name, retriever)

        return f"✅ Successfully inserted {len(documents)} QA pairs into database for {product}."
    except jwt.exceptions.InvalidSignatureError:
        return "Authentication failed: Invalid signature. Please check your encryption key and try again."
    except Exception as exc:
        traceback.print_exc()
        utils.logger.critical("Internal error in update_golden_qa_pairs(): %s", exc)
        return f"Internal error: {str(exc)}"

### /get_conversations/

def get_conversations(product: str, endorsed_by: str, encrypted_key: str):
    try:
        jwt.decode(encrypted_key, AUTHENTICATION_KEY, algorithms="HS256")
        entry_names = os.listdir(LilLisaServerContext.SPEEDICT_FOLDERPATH)
        session_ids = [entry for entry in entry_names if os.path.isdir(os.path.join(LilLisaServerContext.SPEEDICT_FOLDERPATH, entry))]
        useful_conversations = []
        product_enum = PRODUCT.get_product(product)
        for session_id in session_ids:
            llsc = get_llsc(session_id)
            llsc_product = llsc.product
            if product_enum == llsc_product:
                if endorsed_by == "user":
                    endorsements = llsc.user_endorsements
                elif endorsed_by == "expert":
                    endorsements = llsc.expert_endorsements
                else:
                    endorsements = None
                if endorsements:
                    useful_conversations.append(llsc.conversation_history)

        if useful_conversations:
            zip_stream = io.BytesIO()
            with zipfile.ZipFile(zip_stream, "w") as zipf:
                for i, conversation in enumerate(useful_conversations, start=1):
                    filename = f"conversation_{i}.md"
                    conversation_history = ""
                    for poster, message in conversation:
                        conversation_history += f"{poster}: {message}\n"
                    in_memory_file = io.StringIO(conversation_history)
                    zipf.writestr(filename, in_memory_file.getvalue().encode("utf-8"))
            zip_stream.seek(0)
            return zip_stream.getvalue(), "application/zip", "conversations.zip"
        return "No conversations found.", "text/plain", "error.txt"
    except jwt.exceptions.InvalidSignatureError:
        return "Failed signature verification. Unauthorized.", "text/plain", "error.txt"
    except Exception as exc:
        traceback.print_exc()
        utils.logger.critical("Internal error in get_conversations(): %s", exc)
        return f"Internal error: {str(exc)}", "text/plain", "error.txt"

### /rebuild_docs/
def rebuild_docs(encrypted_key: str) -> str:
    try:
        jwt.decode(encrypted_key, AUTHENTICATION_KEY, algorithms="HS256")
        db = lancedb.connect(LANCEDB_FOLDERPATH)
        failed_clone_messages = ""

        product_repos_dict = {
            "IDDM": [
                ("https://github.com/radiantlogic-v8/documentation-new.git", DOCUMENTATION_NEW_VERSIONS),
                ("https://github.com/radiantlogic-v8/documentation-eoc.git", DOCUMENTATION_EOC_VERSIONS),
            ],
            "IDA": [
                ("https://github.com/radiantlogic-v8/documentation-identity-analytics.git", DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS),
                ("https://github.com/radiantlogic-v8/documentation-ia-product.git", DOCUMENTATION_IA_PRODUCT_VERSIONS),
                ("https://github.com/radiantlogic-v8/documentation-ia-selfmanaged.git", DOCUMENTATION_IA_SELFMANAGED_VERSIONS),
            ],
        }

        def find_md_files(directory):
            return [
                os.path.join(root, file)
                for root, _, files in os.walk(directory)
                for file in files
                if file.endswith(".md")
            ]

        def extract_metadata_from_lines(lines):
            metadata = {"title": "", "description": "", "keywords": ""}
            for line in lines:
                if line.startswith("title:"):
                    metadata["title"] = line.split(":", 1)[1].strip()
                elif line.startswith("description:"):
                    metadata["description"] = line.split(":", 1)[1].strip()
                elif line.startswith("keywords:"):
                    metadata["keywords"] = line.split(":", 1)[1].strip()
            return metadata

        def clone_repo(repo_url, target_dir, branch):
            try:
                git.Repo.clone_from(repo_url, target_dir, branch=branch)
                return True
            except Exception:
                return False

        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        node_parser = MarkdownNodeParser()
        reader = MarkdownReader()
        file_extractor = {".md": reader}
        Settings.llm = OpenAI_Llama(model="gpt-3.5-turbo")
        pipeline = IngestionPipeline(transformations=[node_parser])

        os.makedirs(DOCUMENTATION_FOLDERPATH, exist_ok=True)

        excluded_metadata_keys = [
            "file_path",
            "file_name",
            "file_type",
            "file_size",
            "creation_date",
            "last_modified_date",
            "version",
            "github_url"
        ]

        for product, repo_branches in product_repos_dict.items():
            product_dir = os.path.join(DOCUMENTATION_FOLDERPATH, product)
            os.makedirs(product_dir, exist_ok=True)

            max_retries = 5
            all_nodes = []
            for repo_url, branches in repo_branches:
                for branch in branches:
                    repo_name = repo_url.rsplit("/", 1)[-1].replace(".git", "")
                    target_dir = os.path.join(product_dir, repo_name, branch)
                    if os.path.exists(target_dir):
                        shutil.rmtree(target_dir)
                    attempt = 0
                    success = False
                    while attempt < max_retries and not success:
                        success = clone_repo(repo_url, target_dir, branch)
                        if not success:
                            attempt += 1
                            if attempt < max_retries:
                                time.sleep(10)
                            else:
                                failed_clone_messages += f"Max retries reached. Failed to clone {repo_url} ({branch}) into {target_dir}. "

                    md_files = find_md_files(target_dir)
                    for file in md_files:
                        with open(file, "r", encoding="utf-8") as f:
                            first_lines = [next(f).strip() for _ in range(5)]
                        metadata = extract_metadata_from_lines(first_lines)
                        metadata["version"] = branch
                        documents = SimpleDirectoryReader(input_files=[file], file_extractor=file_extractor).load_data()
                        for doc in documents:
                            for label, value in metadata.items():
                                doc.metadata[label] = value
                            file_path = doc.metadata['file_path']
                            relative_path = file_path.replace(f'docs/{product}/', '')
                            github_url = f'https://github.com/radiantlogic-v8/{relative_path}'
                            github_url = github_url.replace(repo_name, f'{repo_name}/blob')
                            doc.metadata['github_url'] = github_url
                        nodes = pipeline.run(documents=documents, in_place=False)
                        for node in nodes:
                            node.excluded_llm_metadata_keys = excluded_metadata_keys
                            node.excluded_embed_metadata_keys = excluded_metadata_keys
                        all_nodes.extend(nodes)

            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            new_nodes_to_add = []
            nodes_to_remove = []
            for node in all_nodes:
                length = len(enc.encode(node.text))
                if length > 7000:
                    nodes_to_remove.append(node)
                    document = Document(text=node.text, metadata=node.metadata)
                    new_nodes = splitter.get_nodes_from_documents(documents=[document])
                    new_nodes_to_add.extend(new_nodes)
            all_nodes = [node for node in all_nodes if node not in nodes_to_remove]
            all_nodes.extend(new_nodes_to_add)

            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
            try:
                db.drop_table(product)
            except Exception:
                utils.logger.exception(f"Table {product} seems to have been deleted. Continuing with insertion process")
            vector_store = LanceDBVectorStore(uri="lancedb", table_name=product, query_type="hybrid")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes=all_nodes[:1], storage_context=storage_context)
            index.insert_nodes(all_nodes[1:])
            retriever = index.as_retriever(similarity_top_k=50)
            update_retriever(product, retriever)

        shutil.rmtree(DOCUMENTATION_FOLDERPATH)
        return "Rebuilt DB successfully!" + failed_clone_messages
    except jwt.exceptions.InvalidSignatureError:
        return "Failed signature verification. Unauthorized."
    except Exception as exc:
        traceback.print_exc()
        utils.logger.critical("Internal error in rebuild_docs(): %s", exc)
        return f"Internal error: {str(exc)}"

# --- Gradio Interface Setup ---
with gr.Blocks(css=custom_css, title="Lil Lisa Server") as demo:
    gr.Markdown("<div class='header'>LilLisa Server</div>")

    # Chatbot Tab
    with gr.Tab("Chatbot"):
        chatbot = gr.Chatbot(elem_classes=["chatbot-container"], label="Conversation")
        session_id_state = gr.State(None)
        locale_dropdown = gr.Dropdown(choices=["en-US", "fr-FR"], label="Locale", value="en-US")
        product_dropdown = gr.Dropdown(choices=["IDA", "IDDM"], label="Product", value="IDA")
        nl_query_input = gr.Textbox(placeholder="Type your query here...", label="Natural Language Query")
        is_expert_answering = gr.Checkbox(label="Is Expert Answering?", value=False)
        with gr.Row():
            send_btn = gr.Button("Send Query")
            clear_btn = gr.Button("Clear Chat")
            endorse_btn = gr.Button("Endorse Conversation")
        endorse_type = gr.Radio(["User", "Expert"], label="Endorsement Type", value="User")

        def chat(nl_query, history, session_id, locale, product, is_expert):
            if not nl_query.strip():
                return history, session_id
            if not session_id or not history:
                session_id = generate_unique_session_id()
            response = invoke(session_id, locale, product, nl_query, is_expert)
            history.append((nl_query, response))
            return history, session_id

        def endorse_conversation(session_id, endorse_type):
            if not session_id:
                return "No active session to endorse."
            is_expert = endorse_type == "Expert"
            return record_endorsement(session_id, is_expert)

        send_btn.click(chat, [nl_query_input, chatbot, session_id_state, locale_dropdown, product_dropdown, is_expert_answering], [chatbot, session_id_state])
        clear_btn.click(lambda: ([], None), None, [chatbot, session_id_state])
        endorse_btn.click(endorse_conversation, [session_id_state, endorse_type], gr.Textbox(label="Endorsement Status"))

    # Admin Tab
    with gr.Tab("Admin Panel"):
        encrypted_key_input = gr.Textbox(label="Encrypted Key", type="password")
        product_admin_dropdown = gr.Dropdown(choices=["IDA", "IDDM"], label="Product", value="IDA")
        endorsed_by_dropdown = gr.Dropdown(choices=["user", "expert"], label="Endorsed By", value="user")
        with gr.Row():
            rebuild_btn = gr.Button("Rebuild Docs")
            get_qa_btn = gr.Button("Get Golden QA Pairs")
            update_qa_btn = gr.Button("Update Golden QA Pairs")
            get_conversations_btn = gr.Button("Get Conversations")
        output_text = gr.Textbox(label="Operation Result")
        output_file = gr.File(label="Download Conversations")

        def handle_rebuild_docs(encrypted_key):
            return rebuild_docs(encrypted_key)

        def handle_get_golden_qa_pairs(product, encrypted_key):
            return get_golden_qa_pairs(product, encrypted_key)

        def handle_update_golden_qa_pairs(product, encrypted_key):
            return update_golden_qa_pairs(product, encrypted_key)

        def handle_get_conversations(product, endorsed_by, encrypted_key):
            content, content_type, filename = get_conversations(product, endorsed_by, encrypted_key)
            if content_type == "application/zip":
                return None, (content, filename)
            return content, None

        rebuild_btn.click(handle_rebuild_docs, [encrypted_key_input], output_text)
        get_qa_btn.click(handle_get_golden_qa_pairs, [product_admin_dropdown, encrypted_key_input], output_text)
        update_qa_btn.click(handle_update_golden_qa_pairs, [product_admin_dropdown, encrypted_key_input], output_text)
        get_conversations_btn.click(handle_get_conversations, [product_admin_dropdown, endorsed_by_dropdown, encrypted_key_input], [output_text, output_file])

# --- Launch the Application ---
if __name__ == "__main__":
    demo.launch(share=False, server_name="localhost", server_port=8080, debug=True)