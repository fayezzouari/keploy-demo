import argparse
import os
from .core import setup_milvus, scrape_repo, embed_and_store, search_snippets, load_milvus_collection, generate_unit_test

def main():
    parser = argparse.ArgumentParser(
        description="Keploy: Code Search CLI",
        epilog="Example usage:\n"
               "  keploy-demo ingest https://github.com/user/repo --token ghp_abc123\n"
               "  keploy-demo search 'authentication middleware' --top-k 5\n"
               "  keploy-demo generate 'validate_token' --query 'JWT validation' --use-llm",
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
        
    # Generate command
    generate_parser = subparsers.add_parser('generate',
        help='Generate unit tests for a function')
    generate_parser.add_argument('function_name',
        help='Name of the function to generate tests for')
    generate_parser.add_argument('--query', default=None,
        help='Search query to find relevant function (if not provided, uses function_name)')
    generate_parser.add_argument('--framework', default='pytest',
        choices=['pytest', 'unittest'],
        help='Test framework to use (default: pytest)')
    generate_parser.add_argument('--top-k', type=int, default=1,
        help='Number of top search results to consider (default: 1)')
    generate_parser.add_argument('--use-llm', action='store_true',
        help='Use LLM (Groq) for enhanced test generation')
    generate_parser.add_argument('--groq-api-key', 
        help='Groq API key (can also be set as GROQ_API_KEY environment variable)')

    args = parser.parse_args()

    if args.command == 'ingest':
        handle_ingest(args)
    elif args.command == 'search':
        handle_search(args)
    elif args.command == 'generate':
        handle_generate(args)

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

def handle_generate(args):
    print(f"🧪 Generating unit tests for function: {args.function_name}")
    
    # Handle LLM setup if requested
    use_llm = args.use_llm
    if use_llm:
        # Set API key from args or environment variable
        if args.groq_api_key:
            os.environ["GROQ_API_KEY"] = args.groq_api_key
        
        if not os.environ.get("GROQ_API_KEY"):
            print("⚠️ Warning: No GROQ_API_KEY found. Please provide it with --groq-api-key or set the GROQ_API_KEY environment variable.")
            print("⚠️ Falling back to rule-based test generation.")
            use_llm = False
        else:
            print("🧠 Using LLM for enhanced test generation...")
    
    # Use function name as query if no specific query provided
    query = args.query if args.query else args.function_name
    
    collection, error = load_milvus_collection()
    if error:
        print(f"❌ {error}")
        return
    
    # First search for the function in the repository
    print(f"🔍 Finding function '{args.function_name}' using query: '{query}'")
    search_results = search_snippets(
        query,
        collection,
        top_k=args.top_k,
        threshold=0.3,
        raw_results=True  # Get raw results for test generation
    )
    
    if not search_results:
        print("❌ No matching functions found")
        return
    
    # Generate unit tests based on the found function
    print("📝 Analyzing function and generating tests...")
    test_code = generate_unit_test(
        search_results,
        args.function_name,
        framework=args.framework,
        use_llm=use_llm
    )
    
    if not test_code:
        print(f"❌ Couldn't generate tests for '{args.function_name}'")
        return
    
    # Save the test code to a file
    filename = f"test_{args.function_name.lower()}.py"
    with open(filename, "w") as f:
        f.write(test_code)
    
    print(f"✅ Generated unit tests in {filename}")
    print("\n📝 Test Preview:")
    print("-" * 40)
    print(test_code[:500] + "..." if len(test_code) > 500 else test_code)
    print("-" * 40)
    
    if use_llm:
        print("\n💡 Tests were enhanced using the Groq LLM for improved quality and coverage.")