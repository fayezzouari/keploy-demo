import argparse
import time
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

import os
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

def search_snippets(query, collection, top_k=3, threshold=0.3,raw_results=False):
    """Search for relevant code snippets."""
    query_embedding = embedder.encode(query, normalize_embeddings=True)
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["snippet", "file_path"]
    )
    if raw_results:
        return [hit for hit in results[0] if hit.distance >= threshold]
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
def generate_unit_test(search_results, function_name, framework='pytest', use_llm=True):
    """
    Generate unit tests for a function found in search results.
    
    Args:
        search_results (list): List of search result objects
        function_name (str): Name of the function to test
        framework (str): Test framework to use ('pytest' or 'unittest')
        use_llm (bool): Whether to enhance tests with LLM
    
    Returns:
        str: The generated unit test code or None if function not found
    """
    import re
    
    # Extract the function definition from the search results
    function_code = None
    file_path = None
    
    for result in search_results:
        snippet = result.entity.snippet
        file_path = result.entity.file_path
        
        # Look for the function definition in the snippet
        pattern = rf"def\s+{re.escape(function_name)}\s*\([^)]*\)(?:\s*->.*?)?:"
        match = re.search(pattern, snippet)
        
        if match:
            # Extract the entire function by finding where it starts and ends
            start_idx = match.start()
            
            # Get the indentation level of the function definition
            def_line = snippet[start_idx:].split('\n')[0]
            base_indent = len(def_line) - len(def_line.lstrip())
            
            # Find the end of the function by looking for a line with same or less indentation
            # that's not a blank line or a comment
            lines = snippet[start_idx:].split('\n')
            end_line_idx = len(lines)
            
            for i, line in enumerate(lines[1:], 1):
                if line.strip() and not line.strip().startswith('#'):
                    curr_indent = len(line) - len(line.lstrip())
                    if curr_indent <= base_indent:
                        end_line_idx = i
                        break
            
            function_code = '\n'.join(lines[:end_line_idx])
            
            # Also try to get context from surrounding code
            context_start = max(0, start_idx - 500)  # Get up to 500 chars before function
            context_before = snippet[context_start:start_idx].strip()
            context_end = min(len(snippet), start_idx + len(function_code) + 500)
            context_after = snippet[start_idx + len(function_code):context_end].strip()
            
            break
    
    if not function_code:
        return None
    
    # If LLM enhancement is requested and API key is available
    if use_llm and os.environ.get("GROQ_API_KEY"):
        try:
            # Initialize LangChain with Groq
            return _generate_tests_with_llm(
                function_code=function_code,
                function_name=function_name,
                file_path=file_path,
                framework=framework,
                context_before=context_before if 'context_before' in locals() else "",
                context_after=context_after if 'context_after' in locals() else ""
            )
        except Exception as e:
            print(f"⚠️ LLM test generation failed: {str(e)}. Falling back to rule-based generation.")
            # Fall back to rule-based generation if LLM fails
    
    # Rule-based generation (original approach as fallback)
    # Determine function signature, parameters, and return type
    signature_match = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:", function_code)
    if not signature_match:
        return None
    
    func_name = signature_match.group(1)
    params_str = signature_match.group(2).strip()
    return_type = signature_match.group(3).strip() if signature_match.group(3) else None
    
    # Parse parameters
    params = []
    if params_str:
        # Handle edge cases like default values, type hints
        current_param = ""
        paren_level = 0
        bracket_level = 0
        
        for char in params_str:
            if char == ',' and paren_level == 0 and bracket_level == 0:
                params.append(current_param.strip())
                current_param = ""
            else:
                current_param += char
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                elif char == '[':
                    bracket_level += 1
                elif char == ']':
                    bracket_level -= 1
        
        if current_param.strip():
            params.append(current_param.strip())
    
    # Process parameters to extract names
    param_names = []
    for param in params:
        # Handle self/cls parameters
        if param in ('self', 'cls'):
            continue
        
        # Strip type hints and default values
        name_part = param.split(':')[0].split('=')[0].strip()
        param_names.append(name_part)
    
    # Extract docstring if present
    docstring_match = re.search(r'"""(.*?)"""', function_code, re.DOTALL)
    docstring = docstring_match.group(1).strip() if docstring_match else ""
    
    # Generate appropriate test cases based on the function signature
    if framework == 'pytest':
        return _generate_pytest(func_name, param_names, return_type, docstring, file_path)
    else:  # unittest
        return _generate_unittest(func_name, param_names, return_type, docstring, file_path)

def _generate_tests_with_llm(function_code, function_name, file_path, framework="pytest", context_before="", context_after=""):
    """Generate unit tests using LangChain with Groq LLM."""
    
    # Initialize the Groq LLM through LangChain
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",  # Using Llama 3 70B model via Groq
        temperature=0.2,  # Low temperature for focused, deterministic output
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    
    # Format the import path
    module_path = file_path.replace('/', '.').replace('\\', '.').rstrip('.py')
    if module_path.startswith('.'):
        module_path = module_path[1:]
    
    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Python developer specializing in test-driven development.
Your task is to generate comprehensive unit tests for a given Python function.
Create realistic, practical test cases that cover:
1. Normal functionality with typical inputs
2. Edge cases and boundary conditions
3. Error handling and validation
4. Any special cases evident from the function's implementation
5. While generating tests, make sure tp detail the script
6. Make sure to write all the unit tests till the end if you have multiple unit tests to generate         

         YOU ARE RESPONSIBLE OF THE GENERATION OF TEST FUNCITONS, TEST VALUES AND EVERY ASPECT OF THE TESTING PROCESS SO MAKE SURE TO MAKE IT RIGOROUS
Focus on writing tests that would catch real bugs. Use realistic input values and meaningful assertions.
Include detailed comments explaining the purpose of each test case."""),
        
        ("user", """Function to test:
```python
{function_code}
```

Context before the function:
```python
{context_before}
```

Context after the function:
```python
{context_after}
```

Please generate {framework} tests for this function.
The function is from module: {module_path}
Import the function using: from {module_path} import {function_name}

Ensure the test file includes:
1. Proper imports
2. Appropriate setup/teardown if needed
3. At least 3 distinct test cases
4. Comprehensive assertions
5. Meaningful comments explaining each test
6. Mocks or fixtures for external dependencies
7. No placeholder TODOs - provide complete implementations""")
    ])
    
    # Create an LLMChain with the prompt template
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # Run the chain
    response = chain.run(
        function_code=function_code,
        function_name=function_name,
        module_path=module_path,
        framework=framework,
        context_before=context_before,
        context_after=context_after
    )
    
    # Extract code blocks if present, otherwise return full response
    import re
    code_blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
    
    if code_blocks:
        # Join all code blocks together
        return "\n\n".join(code_blocks)
    else:
        # If no code blocks, try to extract the test code directly
        return response


def _generate_pytest(func_name, param_names, return_type, docstring, file_path):
    """Generate a pytest test file for the function."""
    module_path = file_path.replace('/', '.').replace('\\', '.').rstrip('.py')
    if module_path.startswith('.'):
        module_path = module_path[1:]
    
    # Extract module name for import
    module_parts = module_path.split('.')
    import_path = '.'.join(module_parts)
    
    # Create multiple test cases
    test_cases = []
    
    # Basic test case
    params_values = []
    params_args = []
    
    for param in param_names:
        if 'id' in param.lower() or 'key' in param.lower():
            params_values.append(f"    {param} = '123'")
            params_args.append(f"{param}={param}")
        elif 'name' in param.lower():
            params_values.append(f"    {param} = 'test_name'")
            params_args.append(f"{param}={param}")
        elif 'list' in param.lower() or 'array' in param.lower() or '[]' in param:
            params_values.append(f"    {param} = [1, 2, 3]")
            params_args.append(f"{param}={param}")
        elif 'dict' in param.lower() or 'map' in param.lower() or '{}' in param:
            params_values.append(f"    {param} = {{'key': 'value'}}")
            params_args.append(f"{param}={param}")
        elif 'bool' in param.lower() or param.lower() in ['flag', 'enabled', 'active']:
            params_values.append(f"    {param} = True")
            params_args.append(f"{param}={param}")
        elif 'int' in param.lower() or 'num' in param.lower():
            params_values.append(f"    {param} = 42")
            params_args.append(f"{param}={param}")
        elif 'float' in param.lower():
            params_values.append(f"    {param} = 3.14")
            params_args.append(f"{param}={param}")
        else:
            params_values.append(f"    {param} = 'test_value'")
            params_args.append(f"{param}={param}")
    
    params_setup = '\n'.join(params_values)
    params_call = ', '.join(params_args)
    
    # Determine expected result based on return type
    if return_type:
        if 'bool' in return_type.lower():
            expected = "True"
        elif 'int' in return_type.lower():
            expected = "42"
        elif 'float' in return_type.lower():
            expected = "3.14"
        elif 'str' in return_type.lower():
            expected = "'expected_result'"
        elif 'list' in return_type.lower() or '[' in return_type:
            expected = "[1, 2, 3]"
        elif 'dict' in return_type.lower() or '{' in return_type:
            expected = "{'key': 'value'}"
        elif 'none' in return_type.lower():
            expected = "None"
        else:
            expected = "result"  # Generic value
    else:
        expected = "result"  # Generic value
    
    # Basic successful test
    test_cases.append(f"""
def test_{func_name}_basic():
    \"\"\"Test basic functionality of {func_name}.\"\"\"
{params_setup}
    
    result = {func_name}({params_call})
    
    # Replace with actual expected result
    assert result is not None
    # assert result == {expected}
""")

    # Edge case test
    test_cases.append(f"""
def test_{func_name}_edge_case():
    \"\"\"Test {func_name} with edge case inputs.\"\"\"
    # Add appropriate edge case values here
{params_setup.replace("'test_value'", "'edge_case'")}
    
    result = {func_name}({params_call})
    
    # Add assertions based on expected edge case behavior
    assert result is not None
""")

    # Error case test if parameters exist
    if param_names:
        test_cases.append(f"""
def test_{func_name}_error_handling():
    \"\"\"Test {func_name} with invalid inputs.\"\"\"
    import pytest
    
    # Choose an appropriate parameter to make invalid
    with pytest.raises(Exception):  # Replace with specific exception if known
        {func_name}({', '.join(['None' for _ in param_names])})
""")

    # Combine all parts
    imports = f"""import pytest
from {import_path} import {func_name}
"""

    test_file = imports + '\n'.join(test_cases)
    
    # Add fixtures if relevant
    if any('db' in param.lower() for param in param_names) or any('conn' in param.lower() for param in param_names):
        fixture = """
@pytest.fixture
def mock_db():
    \"\"\"Provide a mock database connection.\"\"\"
    # Replace with appropriate mock setup
    from unittest.mock import MagicMock
    return MagicMock()
"""
        test_file = imports + fixture + '\n'.join(test_cases).replace('test_basic', 'test_basic(mock_db)')
    
    return test_file

def _generate_unittest(func_name, param_names, return_type, docstring, file_path):
    """Generate a unittest test file for the function."""
    module_path = file_path.replace('/', '.').replace('\\', '.').rstrip('.py')
    if module_path.startswith('.'):
        module_path = module_path[1:]
    
    # Extract module name for import
    module_parts = module_path.split('.')
    import_path = '.'.join(module_parts)
    
    # Create parameter values similar to pytest version
    params_values = []
    params_args = []
    
    for param in param_names:
        if 'id' in param.lower() or 'key' in param.lower():
            params_values.append(f"        self.{param} = '123'")
            params_args.append(f"self.{param}")
        elif 'name' in param.lower():
            params_values.append(f"        self.{param} = 'test_name'")
            params_args.append(f"self.{param}")
        elif 'list' in param.lower() or 'array' in param.lower() or '[]' in param:
            params_values.append(f"        self.{param} = [1, 2, 3]")
            params_args.append(f"self.{param}")
        elif 'dict' in param.lower() or 'map' in param.lower() or '{}' in param:
            params_values.append(f"        self.{param} = {{'key': 'value'}}")
            params_args.append(f"self.{param}")
        elif 'bool' in param.lower() or param.lower() in ['flag', 'enabled', 'active']:
            params_values.append(f"        self.{param} = True")
            params_args.append(f"self.{param}")
        elif 'int' in param.lower() or 'num' in param.lower():
            params_values.append(f"        self.{param} = 42")
            params_args.append(f"self.{param}")
        elif 'float' in param.lower():
            params_values.append(f"        self.{param} = 3.14")
            params_args.append(f"self.{param}")
        else:
            params_values.append(f"        self.{param} = 'test_value'")
            params_args.append(f"self.{param}")
    
    params_setup = '\n'.join(params_values)
    params_call = ', '.join(params_args)
    
    # Generate test class
    test_class = f"""import unittest
from {import_path} import {func_name}


class Test{func_name.title()}(unittest.TestCase):
    \"\"\"Test suite for {func_name} function.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test fixtures before each test method.\"\"\"
{params_setup}
    
    def test_basic_functionality(self):
        \"\"\"Test basic functionality of {func_name}.\"\"\"
        result = {func_name}({params_call})
        
        # Replace with actual expected result
        self.assertIsNotNone(result)
        # Add appropriate assertions based on expected return
    
    def test_edge_case(self):
        \"\"\"Test {func_name} with edge case inputs.\"\"\"
        # Modify parameters for edge case
        result = {func_name}({params_call})
        
        # Add assertions based on expected edge case behavior
        self.assertIsNotNone(result)
"""

    # Add error case test if parameters exist
    if param_names:
        test_class += f"""
    def test_error_handling(self):
        \"\"\"Test {func_name} with invalid inputs.\"\"\"
        # Choose an appropriate parameter to make invalid
        with self.assertRaises(Exception):  # Replace with specific exception if known
            {func_name}({', '.join(['None' for _ in param_names])})
"""

    test_class += """

if __name__ == '__main__':
    unittest.main()
"""
    
    return test_class
