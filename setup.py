from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="keploy-demo",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="CLI for code snippet search using Milvus and GitIngest",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "requests>=2.26.0",
        "beautifulsoup4>=4.10.0",
        "sentence-transformers>=2.2.0",
        "pymilvus>=2.0.0",
        "langchain>=0.0.148",
        "python-dotenv>=0.19.0",
        "tqdm>=4.62.3"
    ],
    entry_points={
        "console_scripts": [
            "keploy-demo=keploy_demo.main:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)