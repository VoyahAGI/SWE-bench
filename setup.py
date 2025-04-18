import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="swebench",
    author="SWE-bench team",
    author_email="support@swebench.com",
    description="The official SWE-bench package - a benchmark for evaluating LMs on software engineering",
    keywords="nlp, benchmark, code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://swebench.com",
    project_urls={
        "Documentation": "https://github.com/swe-bench/SWE-bench",
        "Bug Reports": "http://github.com/swe-bench/SWE-bench/issues",
        "Source Code": "http://github.com/swe-bench/SWE-bench",
        "Website": "https://swebench.com",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "beautifulsoup4",
        "chardet",
        "datasets",
        "docker",
        "ghapi",
        "GitPython",
        "modal",
        "pre-commit",
        "python-dotenv",
        "requests",
        "rich",
        "tenacity",
        "tqdm",
        "unidiff",
    ],
    extras_require={
        "datasets": [
            "protobuf",
            "sentencepiece",
            "tiktoken",
            "transformers",
            "openai",
            "anthropic",
            "jedi",
        ],
        "inference": [
            "torch",
            "anthropic",
            "jedi",
            "openai",
            "peft",
            "protobuf",
            "sentencepiece",
            "tiktoken",
            "transformers",
            "triton",
            "flash_attn",
            "requests",
        ],
        "test": [
            "pytest",
            "pytest-cov",
        ],
        "docs": [
            "mkdocs",
            "mkdocs-material",
            "mkdocs-glightbox",
            "mkdocstrings",
            "mkdocstrings-python",
            "mike",
            "pymdown-extensions",
            "mkdocs-include-markdown-plugin",
            "griffe-pydantic",
        ],
    },
    include_package_data=True,
)
