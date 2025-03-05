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

# --- Custom CSS for Gradio Interface ---
custom_css = """
body {
    background: linear-gradient(135deg, #007bff, #00bcd4);  
    font-family: 'Roboto', sans-serif;
    margin: 0;
    padding: 20px;
}
.chatbot-container {
    background-color: #ffffff;
    border-radius: 15px;
    padding: 20px;
    max-width: 600px;
}
.header {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 20px;
    color: #333;
}
button {
    background-color: #007bff;
    color: white;
    border-radius: 8px;
}
button:hover {
    background-color: #0056b3;
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
        return "ok"
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
        return "No QA pairs found."
    except jwt.exceptions.InvalidSignatureError:
        return "Failed signature verification. Unauthorized."
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

        return "Successfully inserted QA pairs into DB."
    except jwt.exceptions.InvalidSignatureError:
        return "Failed signature verification. Unauthorized."
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
    with gr.Tab("Admin"):
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