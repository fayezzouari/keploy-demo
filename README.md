# Keploy GSoC 2025 Project Demo: GitHub Snippet Search with GitIngest

<div align="center">
  <img src="assets/keploy-logo-dark.svg" width="400">
</div>

## Overview

This demo is part of a **Google Summer of Code (GSoC) 2025** project proposal for **Keploy**, an API testing platform. It allows users to ingest a GitHub repository using GitIngest and search for code snippets (e.g., JWT authentication in a web app). Built with Gradio, Milvus, and sentence-transformers, it supports semantic search over repository content.

The project also features a powerful CLI tool for repository ingestion, code search, and **unit test generation** from retrieved snippets (e.g., generating tests for JWT authentication code).

---

## Features

- **Ingest Repositories**: Fetch GitHub repo content via GitIngest.
- **Semantic Search**: Query snippets (e.g., "JWT authentication" in a web app).
- **File Paths & Scores**: Results include file paths and cosine similarity scores.
- **Unit Test Generation**: Generate tests for specific functions found in the repository.
- **Multiple Interfaces**: Both web UI (Gradio) and CLI for flexibility.
- **LLM Integration**: Optional enhanced test generation using Groq.
- **Framework Support**: Choose between pytest and unittest frameworks.

---

## Installation

### Prerequisites
- Docker (for Milvus)
- Python 3.8+
- Git

### Step 1: Install Milvus
Milvus is used for vector storage and search.
1. Download the script:
   
```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
```

2. Start Milvus:
   
```bash
bash standalone_embed.sh start
```

   - Runs on localhost:19530.
   - Verify: docker ps (look for milvus-standalone).

### Step 2: Clone the Repository
```bash
git clone https://github.com/fayezzouari/keploy-demo.git
cd keploy-demo
```

### Step 3: Set Up Python Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install gradio==3.50.2 requests beautifulsoup4 sentence-transformers pymilvus langchain
```

For LLM-enhanced test generation:
```bash
pip install groq  # For Groq LLM integration
```

### Step 4: Run the App
```bash
# Web UI
python main.py

# CLI mode (see Usage section)
python -m keploy_demo <command> [options]
```

- Web UI access at http://localhost:7860.

---

## Usage

### Web Interface

#### Step 1: Ingest a Repository
1. Enter a GitHub URL (e.g., https://github.com/webapp/website for a web app).
2. Add a token for private repos (optional).
3. Click "Ingest Repository".
   - **Output**: "Stored 120 chunks in Milvus. **Ready to search!**"
   - Status: "Ingested: https://github.com/webapp/website"

#### Step 2: Search Snippets
1. Enter a query (e.g., "JWT authentication").
2. Click "Search".
   - **Output**:
     
**File**: docs/auth.md
     **Snippet**:
```plaintext
     --- File: docs/auth.md ---
     To secure APIs, we use JWT tokens for authentication...
```
     
**Similarity Score**: 0.8923

   - If no matches: "No snippets found above similarity threshold."

#### Step 3: Clear State (Optional)
- Click "Clear State" to reset for a new repo.

### Command Line Interface

The CLI provides three main commands: `ingest`, `search`, and `generate`.

#### Ingest a Repository
```bash
python -m keploy_demo ingest https://github.com/user/repo --token ghp_abc123
```

#### Search for Code Snippets
```bash
python -m keploy_demo search "authentication middleware" --top-k 5 --threshold 0.3
```

#### Generate Unit Tests
```bash
# Basic test generation using function name
python -m keploy_demo generate validate_token --framework pytest

# Advanced test generation with custom search query
python -m keploy_demo generate validate_token --query "JWT validation" --framework pytest

# Enhanced test generation using LLM
python -m keploy_demo generate validate_token --use-llm --groq-api-key your_api_key
```

---

## CLI Reference

```
usage: keploy-demo [-h] {ingest,search,generate} ...

Keploy: Code Search CLI

optional arguments:
  -h, --help            show this help message and exit

commands:
  {ingest,search,generate}
    ingest              Ingest a GitHub repository
    search              Search code snippets
    generate            Generate unit tests for a function

Example usage:
  keploy-demo ingest https://github.com/user/repo --token ghp_abc123
  keploy-demo search 'authentication middleware' --top-k 5
  keploy-demo generate 'validate_token' --query 'JWT validation' --use-llm
```

### Ingest Command
```
usage: keploy-demo ingest [-h] [--token TOKEN] repo_url

positional arguments:
  repo_url       GitHub repository URL (e.g. https://github.com/user/repo)

optional arguments:
  -h, --help     show this help message and exit
  --token TOKEN  GitHub access token for private repositories
```

### Search Command
```
usage: keploy-demo search [-h] [--top-k TOP_K] [--threshold THRESHOLD] query

positional arguments:
  query                 Search query (e.g. "JWT authentication")

optional arguments:
  -h, --help            show this help message and exit
  --top-k TOP_K         Number of results to return (default: 3)
  --threshold THRESHOLD Similarity threshold (default: 0.3)
```

### Generate Command
```
usage: keploy-demo generate [-h] [--query QUERY] [--framework {pytest,unittest}]
                           [--top-k TOP_K] [--use-llm] [--groq-api-key GROQ_API_KEY]
                           function_name

positional arguments:
  function_name         Name of the function to generate tests for

optional arguments:
  -h, --help            show this help message and exit
  --query QUERY         Search query to find relevant function (if not provided, uses function_name)
  --framework {pytest,unittest}
                        Test framework to use (default: pytest)
  --top-k TOP_K         Number of top search results to consider (default: 1)
  --use-llm             Use LLM (Groq) for enhanced test generation
  --groq-api-key GROQ_API_KEY
                        Groq API key (can also be set as GROQ_API_KEY environment variable)
```

---

## Troubleshooting
- **Milvus Fails**: Check `docker logs milvus-standalone` for errors (e.g., port conflicts on 19530).
- **Ingestion Fails**: Ensure the repo URL is valid; use a token for private repos.
- **Search Fails**: Ingest first; check "Debug - is_ingested: True".
- **Gradio Issues**: Upgrade to `gradio==3.50.2` if UI fails.
- **LLM Integration**: Set `GROQ_API_KEY` environment variable or use the `--groq-api-key` option.
- **Missing Package**: If you see module errors, ensure you've installed all required packages.

---

## Future Work
- Expand test generation capabilities to handle more complex functions
- Support multiple repositories simultaneously
- Add a loading indicator in the web UI
- Implement test execution and results visualization
- Integrate with more LLM providers beyond Groq

---

## Contributing
Contributions are welcome! Fork, branch, and submit a PR. Focus areas: unit test generation, UI enhancements, CLI improvements.

---

## License
MIT License (see `LICENSE`).

---

## Contact
- GitHub: [Fayez Zouari](https://github.com/fayezzouari)
- Email: fayez.zouari@insat.ucar.tn

This demo is a step toward enhancing UTG with Keploy for GSoC 2025!