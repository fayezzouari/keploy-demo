# Keploy GSoC 2025 Project Demo: GitHub Snippet Search with GitIngest

## Overview

This demo is part of a **Google Summer of Code (GSoC) 2025** project proposal for **Keploy**, an API testing platform. It allows users to ingest a GitHub repository using GitIngest and search for code snippets (e.g., JWT authentication in a web app). Built with Gradio, Milvus, and sentence-transformers, it supports semantic search over repository content.

**Note**: This is under active development. I plan to add **unit test generation** for retrieved snippets (e.g., generating tests for JWT authentication code) in the next phase.

---

## Features

- **Ingest Repositories**: Fetch GitHub repo content via GitIngest.
- **Semantic Search**: Query snippets (e.g., "JWT authentication" in a web app).
- **File Paths & Scores**: Results include file paths and cosine similarity scores.
- **Two-Step Workflow**: Ingest once, search multiple times.
- **Clear State**: Reset the app for a new repository.

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
   - Runs on `localhost:19530`.
   - Verify: `docker ps` (look for `milvus-standalone`).

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

### Step 4: Run the App
```bash
python main.py
```
- Access at `http://localhost:7860`.

---

## Usage

### Step 1: Ingest a Repository
1. Enter a GitHub URL (e.g., `https://github.com/webapp/website` for a web app).
2. Add a token for private repos (optional).
3. Click "Ingest Repository".
   - **Output**: "Stored 120 chunks in Milvus. **Ready to search!**"
   - Status: "Ingested: https://github.com/webapp/website"

### Step 2: Search Snippets
1. Enter a query (e.g., "JWT authentication").
2. Click "Search".
   - **Output**:
     ```
     **File**: docs/auth.md
     **Snippet**:
     ```plaintext
     --- File: docs/auth.md ---
     To secure APIs, we use JWT tokens for authentication...
     ```
     **Similarity Score**: 0.8923

   - If no matches: "No snippets found above similarity threshold."

### Step 3: Clear State (Optional)
- Click "Clear State" to reset for a new repo.

---

## Troubleshooting
- **Milvus Fails**: Check `docker logs milvus-standalone` for errors (e.g., port conflicts on 19530).
- **Ingestion Fails**: Ensure the repo URL is valid; use a token for private repos.
- **Search Fails**: Ingest first; check "Debug - is_ingested: True".
- **Gradio Issues**: Upgrade to `gradio==3.50.2` if UI fails.

---

## Future Work
- Add **unit test generation** (e.g., tests for JWT authentication code).
- Support multiple repos.
- Add a loading indicator.

---

## Contributing
Contributions are welcome! Fork, branch, and submit a PR. Focus areas: unit test generation, UI enhancements.

---

## License
MIT License (see `LICENSE`).

---

## Contact
- GitHub: [Fayez Zouari](https://github.com/fayezzouari)
- Email: fayez.zouari@insat.ucar.tn

This demo is a step toward enhancing UTG with Keploy for GSoC 2025!