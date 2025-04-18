import argparse
from .core import setup_milvus, scrape_repo, embed_and_store, search_snippets, load_milvus_collection

def main():
    parser = argparse.ArgumentParser(
        description="Keploy: Code Search CLI",
        epilog="Example usage:\n"
               "  keploy-demo ingest https://github.com/user/repo --token ghp_abc123\n"
               "  keploy-demo search 'authentication middleware' --top-k 5",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', 
        help='Ingest a GitHub repository')
    ingest_parser.add_argument('repo_url', 
        help='GitHub repository URL (e.g. https://github.com/user/repo)')
    ingest_parser.add_argument('--token', 
        help='GitHub access token for private repositories')

    # Search command
    search_parser = subparsers.add_parser('search', 
        help='Search code snippets')
    search_parser.add_argument('query', 
        help='Search query (e.g. "JWT authentication")')
    search_parser.add_argument('--top-k', type=int, default=3,
        help='Number of results to return (default: 3)')
    search_parser.add_argument('--threshold', type=float, default=0.3,
        help='Similarity threshold (default: 0.3)')

    args = parser.parse_args()

    if args.command == 'ingest':
        handle_ingest(args)
    elif args.command == 'search':
        handle_search(args)

def handle_ingest(args):
    print(f"🚀 Starting ingestion of {args.repo_url}")
    collection = setup_milvus()
    
    if isinstance(collection, str):
        print(f"❌ Error: {collection}")
        return
    
    print("🔍 Scraping repository content...")
    content = scrape_repo(args.repo_url, args.token)
    
    if content.startswith(("Invalid", "Error", "Failed")):
        print(f"❌ Error: {content}")
        return
    
    print("📦 Storing embeddings in Milvus...")
    result = embed_and_store(content, collection)
    print(f"✅ {result}")

def handle_search(args):
    print(f"🔎 Searching for: {args.query}")
    collection, error = load_milvus_collection()
    
    if error:
        print(f"❌ {error}")
        return
    
    results = search_snippets(
        args.query,
        collection,
        top_k=args.top_k,
        threshold=args.threshold
    )
    
    print("\n📄 Search Results:")
    print(results if results else "No matching results found")