import argparse
import time
import requests
from bs4 import BeautifulSoup
from keploy_demo.repo_ingestion import fetch_download_link, parse_repository_content
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# Constants
COLLECTION_NAME = "repo_snippets"
DIMENSION = 384
BASE_URL = "https://gitingest.com"
embedder = SentenceTransformer('all-MiniLM-L6-v2')

class GitIngestError(Exception):
    """Custom exception for GitIngest-related errors"""
    pass

def setup_milvus():
    """Initialize and return a new Milvus collection with error handling"""
    try:
        # Verify connection first
        connections.connect("default", host="localhost", port="19530")
        if not connections.has_connection("default"):
            raise Exception("Failed to establish connection to Milvus")
    except Exception as e:
        return f"Milvus connection failed: {str(e)}. Verify Docker is running."

    # Check existing collection with better error handling
    try:
        if utility.has_collection(COLLECTION_NAME):
            collection = Collection(COLLECTION_NAME)
            if collection.is_empty:
                collection.drop()
            else:
                # Release all loaded data first
                collection.release()
                utility.drop_collection(COLLECTION_NAME)
    except Exception as e:
        return f"Collection cleanup failed: {str(e)}"

    # Create new collection with explicit parameters
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="snippet", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1024)
    ]
    
    schema = CollectionSchema(fields, description="Code snippets")
    collection = Collection(COLLECTION_NAME, schema)

    # Create index with safer parameters
    index_params = {
        "metric_type": "COSINE",  # Must match search metric
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    
    try:
        collection.create_index("embedding", index_params)
        collection.load()
        return collection
    except Exception as e:
        return f"Index creation failed: {str(e)}"

def load_milvus_collection():
    """Improved collection loading with connection verification"""
    try:
        # Verify active connection
        if not connections.has_connection("default"):
            connections.connect("default", host="localhost", port="19530")
        
        # Wait for connection to stabilize
        time.sleep(1)
        
        if not utility.has_collection(COLLECTION_NAME):
            return None, "No collection found - run 'ingest' first"
            
        collection = Collection(COLLECTION_NAME)
        collection.load()
        return collection, None
    except Exception as e:
        return None, f"Connection failed: {str(e)}. Verify Milvus is running."
    
def scrape_repo(repo_url, github_token=""):
    """Scrape a GitHub repository using GitIngest with improved format detection"""
    try:
        if not repo_url.startswith("https://github.com/"):
            return f"Invalid GitHub URL: {repo_url}"
        
        repo_name = repo_url.replace("https://github.com/", "").strip("/")
        if not repo_name:
            return "Empty repository name"
        
        if not re.match(r"^[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+$", repo_name):
            return f"Invalid repository name format: {repo_name}"
        
        # Create a new session with better headers
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        })
        
        try:
            # Submit the form directly
            payload = {
                "input_text": repo_name,
                "pattern_type": "exclude",
                "pattern": "",
                "max_file_size": "512"
            }
            
            print(f"Submitting repository: {repo_name} to GitIngest")
            response = session.post(
                BASE_URL,
                data=payload,
                headers={"Referer": BASE_URL},
                timeout=30
            )
            response.raise_for_status()
            
            # Parse response for download link
            soup = BeautifulSoup(response.text, "html.parser")
            download_link_tag = soup.find("a", href=lambda href: href and "download" in href)
            
            if not download_link_tag:
                return "Download link not found in GitIngest response"
            
            # Extract the download link
            download_link = download_link_tag["href"]
            if not download_link.startswith("http"):
                download_link = BASE_URL + download_link
            
            print(f"Found download link: {download_link}")
            
            # Download the repository content
            response = session.get(download_link, timeout=30)
            response.raise_for_status()
            content = response.text
            
            # Try several patterns to identify files in the content
            file_contents = []
            
            # Pattern 1: Traditional separator pattern
            traditional_matches = re.finditer(r"^(={20,})\s*File:\s*(.*?)\s*\1", content, re.MULTILINE | re.DOTALL)
            
            # Pattern 2: Markdown style headers
            markdown_matches = re.finditer(r"^(#{1,6})\s+(.*?\.(?:py|js|java|cpp|h|c|go|rb|rs|php|html|css|tsx?|jsx?))\s*$", 
                                      content, re.MULTILINE)
            
            # Pattern 3: Filename followed by content
            filename_matches = re.finditer(r"^(.*?\.(?:py|js|java|cpp|h|c|go|rb|rs|php|html|css|tsx?|jsx?))\s*[:=]\s*$",
                                     content, re.MULTILINE)
            
            # Pattern 4: Special formatted filename blocks (like <file:path/to/file.py>)
            special_matches = re.finditer(r"[<\[](file|path):([^>\]]+\.[a-zA-Z]+)[>\]]", content)
            
            patterns_tried = []
            
            # Try traditional separator
            file_blocks = []
            for match in traditional_matches:
                separator = match.group(1)
                separator_pattern = rf"^{re.escape(separator)}\s*File:\s*(.*?)\s*{re.escape(separator)}"
                file_blocks = list(re.finditer(separator_pattern, content, re.MULTILINE | re.DOTALL))
                if file_blocks:
                    patterns_tried.append("traditional")
                    break
            
            if file_blocks:
                for i, match in enumerate(file_blocks):
                    filename = match.group(1).strip()
                    content_start = match.end()
                    content_end = file_blocks[i+1].start() if i < len(file_blocks)-1 else len(content)
                    file_content = content[content_start:content_end].strip()
                    file_contents.append((filename, file_content))
            
            # If no files found, split the content by common patterns
            if not file_contents:
                # Try a general approach by splitting on potential file markers
                pattern = r"(^|\n)(?:---+\s*|\*{3,}\s*|#{3,}\s*|={3,}\s*)?(?:File|PATH|FILENAME)?\s*[:\-=]?\s*([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]+)(?:\s*[:\-=]|\s*$)"
                general_matches = list(re.finditer(pattern, content, re.IGNORECASE))
                
                if general_matches:
                    patterns_tried.append("general")
                    for i, match in enumerate(general_matches):
                        filename = match.group(2).strip()
                        content_start = match.end()
                        content_end = general_matches[i+1].start() if i < len(general_matches)-1 else len(content)
                        file_content = content[content_start:content_end].strip()
                        file_contents.append((filename, file_content))
            
            # Last resort: try to find filenames and split content
            if not file_contents:
                patterns_tried.append("filename_extraction")
                filename_pattern = r"(?:^|\n)([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]{1,5})(?:\s*:|\s*$)"
                filenames = list(re.finditer(filename_pattern, content))
                
                if filenames:
                    for i, match in enumerate(filenames):
                        filename = match.group(1).strip()
                        content_start = match.end()
                        content_end = filenames[i+1].start() if i < len(filenames)-1 else len(content)
                        file_content = content[content_start:content_end].strip()
                        if len(file_content) > 10:  # Ensure we have meaningful content
                            file_contents.append((filename, file_content))
            
            # If we still have no content, see if it looks like a single file
            if not file_contents and len(content) > 100:
                patterns_tried.append("single_file")
                # Extract likely filename from URL or repo name
                parts = repo_name.split('/')
                if len(parts) >= 2:
                    repo_dir = parts[-1]
                    likely_extension = ".py" if "python" in content.lower() or "def " in content else ".js"
                    filename = f"{repo_dir}/main{likely_extension}"
                    file_contents.append((filename, content))
            
            if not file_contents:
                # Save content for debugging
                with open("gitingest_content.txt", "w", encoding="utf-8") as f:
                    f.write(content[:5000])  # Save more content
                return f"No file patterns identified. Tried: {', '.join(patterns_tried)}. See gitingest_content.txt"
                
            # Format the output as before
            formatted_output = "\n".join([f"--- File: {fn} ---\n{cnt}" for fn, cnt in file_contents])
            
            return formatted_output
            
        except requests.RequestException as e:
            return f"Network error: {str(e)}"
            
    except Exception as e:
        import traceback
        print(f"Unexpected error in scrape_repo: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return f"Unexpected error: {str(e)}"
    
    
def embed_and_store(content, collection):
    """Split content into chunks and store embeddings in Milvus."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    print(content)
    chunks = text_splitter.split_text(content)
    
    embeddings, snippets, file_paths = [], [], []
    for chunk in chunks:
        file_path = next((line.split("--- File: ")[1].split(" ---")[0] 
                       for line in chunk.split("\n") if line.startswith("--- File: ")), "Unknown")
        embeddings.append(embedder.encode(chunk, normalize_embeddings=True))
        snippets.append(chunk)
        file_paths.append(file_path)
    
    collection.insert([embeddings, snippets, file_paths])
    return f"Ingested {len(chunks)} code snippets."

def search_snippets(query, collection, top_k=3, threshold=0.3):
    """Search for relevant code snippets."""
    query_embedding = embedder.encode(query, normalize_embeddings=True)
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["snippet", "file_path"]
    )
    
    output = []
    for hit in results[0]:
        if hit.distance < threshold:
            continue
        output.append(
            f"File: {hit.entity.file_path}\n"
            f"Similarity: {hit.distance:.2f}\n"
            f"Snippet:\n{hit.entity.snippet}\n{'-'*40}"
        )
    return "\n\n".join(output) if output else "No relevant results found."

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Keploy: Code Search CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest a GitHub repository')
    ingest_parser.add_argument('repo_url', help='GitHub repository URL')
    ingest_parser.add_argument('--token', help='GitHub access token for private repos')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search code snippets')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top-k', type=int, default=3,
                             help='Number of results to return (default: 3)')
    search_parser.add_argument('--threshold', type=float, default=0.3,
                             help='Similarity threshold (default: 0.3)')

    args = parser.parse_args()

    if args.command == 'ingest':
        # Repository ingestion workflow
        collection = setup_milvus()
        if isinstance(collection, str):
            print(f"Error: {collection}")
            return
        
        content = scrape_repo(args.repo_url, args.token)
        if content.startswith(("Invalid", "Error", "Failed")):
            print(f"Error: {content}")
            return
        
        result = embed_and_store(content, collection)
        print(f"Success: {result}")

    elif args.command == 'search':
        # Code search workflow
        collection, error = load_milvus_collection()
        if error:
            print(f"Error: {error}")
            return
        
        results = search_snippets(args.query, collection, args.top_k, args.threshold)
        print(f"Search results for '{args.query}':\n\n{results}")

if __name__ == "__main__":
    main()