import gradio as gr
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Milvus setup
COLLECTION_NAME = "repo_snippets"
DIMENSION = 384
BASE_URL = "https://gitingest.com"

def setup_milvus():
    try:
        connections.connect("default", host="localhost", port="19530")
    except Exception as e:
        return f"Failed to connect to Milvus: {str(e)}"
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="snippet", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1024)
    ]
    schema = CollectionSchema(fields, description="Code snippets from repository")
    
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
        collection.drop()
    collection = Collection(COLLECTION_NAME, schema)
    
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    return collection

def scrape_repo(repo_url, github_token=""):
    try:
        if not repo_url.startswith("https://github.com/"):
            return "Invalid repository URL. Use format: https://github.com/owner/repo"
        repo_name = repo_url.replace("https://github.com/", "").strip("/")
    except:
        return "Error processing URL."

    payload = {
        "input_text": repo_name,
        "max_file_size": 243,
        "pattern_type": "exclude",
        "pattern": "",
    }
    session = requests.Session()
    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    
    try:
        response = session.post(BASE_URL, data=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            return f"Failed to fetch GitIngest form: {response.status_code}"
        
        soup = BeautifulSoup(response.text, "html.parser")
        download_link_tag = soup.find("a", href=lambda href: href and "download" in href)
        
        if not download_link_tag:
            return "Failed to find download link in GitIngest response."
        
        download_link = BASE_URL + download_link_tag["href"]
    except requests.RequestException as e:
        return f"Error fetching GitIngest form: {str(e)}"

    try:
        response = session.get(download_link, headers=headers, stream=True, timeout=30)
        if response.status_code != 200:
            return f"Failed to download text file: {response.status_code}"
        
        content = response.text
        if not content.strip():
            return "No content retrieved from GitIngest."
    except requests.RequestException as e:
        return f"Error downloading text file: {str(e)}"

    # print("GitIngest content (first 1000 chars):\n", content[:1000])

    separator_pattern = r'={40,}\n\s*(?:FILE|File):\s*(.*?)\s*\n={40,}'
    matches = list(re.finditer(separator_pattern, content, re.DOTALL | re.IGNORECASE))
    
    if not matches:
        return f"No file separators found in GitIngest content. Content sample:\n{content[:500]}"
    
    file_contents = []
    filenames = [match.group(1).strip() for match in matches]
    start_positions = [match.start() for match in matches]
    
    for i in range(len(start_positions)):
        filename = filenames[i]
        match_text = matches[i].group(0)
        content_start = start_positions[i] + len(match_text)
        content_end = start_positions[i + 1] if i < len(start_positions) - 1 else len(content)
        file_content = content[content_start:content_end].strip()
        file_contents.append((filename, file_content))
    
    all_content = ""
    for filename, file_content in file_contents:
        all_content += f"\n--- File: {filename} ---\n{file_content}\n"
    
    return all_content

def     embed_and_store(content, collection):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(content)
    
    embeddings = []
    snippet_texts = []
    file_paths = []
    
    for chunk in chunks:
        file_path = "Unknown"
        for line in chunk.split("\n"):
            if line.startswith("--- File: "):
                file_path = line.replace("--- File: ", "").replace(" ---", "").strip()
                break
        
        embeddings.append(embedder.encode(chunk.strip(), normalize_embeddings=True))
        snippet_texts.append(chunk.strip())
        file_paths.append(file_path)

    entities = [
        embeddings,
        snippet_texts,
        file_paths
    ]
    collection.insert(entities)
    return f"Stored {len(chunks)} chunks in Milvus.", collection

def search_snippets(query, collection, top_k=3, threshold=0.3):
    if not collection:
        return "No repository ingested yet. Please ingest a repository first."
    
    query_embedding = embedder.encode(query, normalize_embeddings=True)
    
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["snippet", "file_path"]
    )
    
    output = ""
    for hits in results:
        for hit in hits:
            similarity = hit.distance
            if similarity < threshold:
                continue
            output += f"**File**: {hit.entity.get('file_path')}\n"
            output += f"**Snippet**:\n```plaintext\n{hit.entity.get('snippet')}\n```\n"
            output += f"**Similarity Score**: {similarity:.4f}\n\n"
    return output if output else "No snippets found above similarity threshold."

def ingest_repo(repo_url, github_token, is_ingested):
    print(f"Before ingestion - is_ingested: {is_ingested}")
    try:
        collection = setup_milvus()
        if isinstance(collection, str):
            print(f"Failed to setup Milvus - is_ingested: {is_ingested}")
            return collection, None, False, "No repository ingested."
        
        content = scrape_repo(repo_url, github_token)
  
        result, collection = embed_and_store(content, collection)
        print(f"Ingestion complete: {result} - Setting is_ingested to True")
        return (
            f"{result}\n\n**Ready to search!** Enter a query and click 'Search'.",
            collection,
            True,
            f"Ingested: {repo_url}"
        )
    except Exception as e:
        print(f"Error during ingestion: {str(e)} - is_ingested: {is_ingested}")
        return f"Error during ingestion: {str(e)}", None, False, "No repository ingested."

def search_query(query, collection, is_ingested):
    print(f"Search query - is_ingested: {is_ingested}, collection: {collection is not None}")
    if not is_ingested:
        return "No repository ingested yet (is_ingested=False). Please ingest a repository first."
    if not collection:
        return "No repository ingested yet (collection=None). Please ingest a repository first."
    if not query:
        return "Please enter a search query."
    try:
        result = search_snippets(query, collection)
        return result
    except Exception as e:
        return f"Error during search: {str(e)}"

def clear_state():
    return None, False, "No repository ingested.", "", ""

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# GitHub Repository Snippet Search with GitIngest")
    
    collection_state = gr.State(value=None)
    is_ingested = gr.State(value=False)
    status_message = gr.State(value="No repository ingested.")
    
    gr.Markdown("### Status")
    status_display = gr.Markdown(value="No repository ingested.")
    
    gr.Markdown("## Step 1: Ingest Repository via GitIngest")
    repo_url = gr.Textbox(label="Repository URL", placeholder="https://github.com/owner/repo")
    github_token = gr.Textbox(label="GitHub Token (optional)", type="password", placeholder="Enter token for private repos")
    with gr.Row():
        ingest_button = gr.Button("Ingest Repository")
        clear_button = gr.Button("Clear State")
    ingest_output = gr.Markdown(label="Ingestion Output")
    debug_state = gr.Markdown("Debug - is_ingested: False")
    
    gr.Markdown("## Step 2: Search Snippets")
    query = gr.Textbox(label="Search Query", placeholder="e.g., JWT Authentication")
    gr.Markdown("**Note**: Ingest a repository before searching.")
    search_button = gr.Button("Search")
    search_output = gr.Markdown(label="Search Output")
    
    def update_debug_state(is_ingested):
        return f"Debug - is_ingested: {is_ingested}"

    ingest_button.click(
        fn=ingest_repo,
        inputs=[repo_url, github_token, is_ingested],
        outputs=[ingest_output, collection_state, is_ingested, status_message]
    ).then(
        fn=lambda status: status,
        inputs=[status_message],
        outputs=[status_display]
    ).then(
        fn=update_debug_state,
        inputs=[is_ingested],
        outputs=[debug_state]
    )
    
    search_button.click(
        fn=search_query,
        inputs=[query, collection_state, is_ingested],
        outputs=search_output
    )
    
    clear_button.click(
        fn=clear_state,
        inputs=[],
        outputs=[collection_state, is_ingested, status_message, ingest_output, search_output]
    ).then(
        fn=lambda: "No repository ingested.",
        inputs=[],
        outputs=[status_display]
    ).then(
        fn=lambda: "Debug - is_ingested: False",
        inputs=[],
        outputs=[debug_state]
    )

demo.launch()