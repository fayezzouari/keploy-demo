import os
import re
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Tuple, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://gitingest.com"
VALID_REPO_PATTERN = r"^[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+$"

class GitIngestError(Exception):
    """Custom exception for GitIngest-related errors"""
    pass

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_download_link(repository_name: str) -> Tuple[str, requests.Session]:
    """Fetch download link from GitIngest with validation and retries (synchronous version)"""
    try:
        if not re.match(VALID_REPO_PATTERN, repository_name):
            raise ValueError(f"Invalid repository name format: {repository_name}")

        # First get the CSRF token
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

        # Get initial page to obtain CSRF token
        response = session.get(BASE_URL)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        csrf_token = soup.find("input", {"name": "csrf_token"}).get("value", "")
        
        if not csrf_token:
            raise GitIngestError("CSRF token not found in initial request")

        payload = {
            "csrf_token": csrf_token,
            "input_text": repository_name,
            "max_file_size": 512,  # Increased from 243 for better compatibility
            "pattern_type": "exclude",
            "pattern": "",
        }

        response = session.post(
            BASE_URL,
            data=payload,
            headers={"Referer": BASE_URL}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        download_link_tag = soup.find("a", href=lambda href: href and "download" in href)
        
        if not download_link_tag:
            error_msg = soup.find("div", class_="error-message")
            if error_msg:
                raise GitIngestError(f"GitIngest error: {error_msg.text.strip()}")
            raise GitIngestError("Download link not found in response")

        download_link = BASE_URL + download_link_tag["href"]
        logger.info(f"Successfully obtained download link: {download_link}")
        return download_link, session

    except requests.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        raise GitIngestError(f"Network error occurred: {str(e)}")
    

def download_and_save_repository(repository_name: str) -> str:
    """Download repository with validation and proper path handling"""
    try:
        download_url, session = fetch_download_link(repository_name)
        safe_repo_name = re.sub(r"[^\w-]", "_", repository_name)
        save_dir = os.path.join(os.getcwd(), "parsed_repositories", safe_repo_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "code.txt")

        response = session.get(download_url, stream=True, timeout=30)
        response.raise_for_status()

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)

        logger.info(f"Repository successfully saved to: {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise GitIngestError(f"Failed to download repository: {str(e)}")

def parse_repository_content(content: str) -> List[Tuple[str, str]]:
    """Improved content parser with dynamic separator detection"""
    try:
        # Find the separator pattern dynamically
        first_separator = re.search(r"^(=+)\s*File:", content, re.MULTILINE)
        if not first_separator:
            return []

        separator = first_separator.group(1)
        separator_pattern = rf"^{re.escape(separator)}\s*File:\s*(.*?)\s*{re.escape(separator)}"
        
        matches = list(re.finditer(separator_pattern, content, re.MULTILINE | re.DOTALL))
        if not matches:
            return []

        file_contents = []
        for i, match in enumerate(matches):
            filename = match.group(1).strip()
            content_start = match.end()
            content_end = matches[i+1].start() if i < len(matches)-1 else len(content)
            file_content = content[content_start:content_end].strip()
            
            # Clean up common artifacts
            file_content = re.sub(r"^\n+", "", file_content)
            file_content = re.sub(r"\n+$", "", file_content)
            
            file_contents.append((filename, file_content))

        return file_contents

    except Exception as e:
        logger.error(f"Parsing error: {str(e)}")
        return []

async def process_repository(repository_name: str) -> List[Dict[str, str]]:
    """Main processing pipeline with enhanced validation"""
    try:
        # Download repository
        file_path = await download_and_save_repository(repository_name)
        
        # Read and parse content
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        
        file_contents = parse_repository_content(content)
        
        # Filter and validate files
        valid_files = [
            (fn, cnt) for fn, cnt in file_contents 
            if len(cnt) > 100 and not fn.endswith(('.png', '.jpg', '.jpeg', '.gif'))
        ]

        # Split text with code-aware splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\nfunction ", "\nclass ", "\ndef ", "\n//", "\n#", "\n\n", " "]
        )

        chunks = []
        for filename, content in valid_files:
            for chunk in text_splitter.split_text(content):
                chunks.append({
                    "documentSource": f"github:{repository_name}",
                    "documentURL": f"https://github.com/{repository_name}/blob/main/{filename}",
                    "documentContent": chunk,
                    "filePath": filename
                })

        logger.info(f"Processed {len(chunks)} chunks from {len(valid_files)} files")
        return chunks

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise