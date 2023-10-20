"""
Prompt & model versioning on the cloud, built for developers.
"""
from setuptools import setup, find_namespace_packages

# Read README.md for the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="promptmodel",
    version="0.0.5",
    packages=find_namespace_packages(),
    entry_points={
        "console_scripts": [
            "prompt = promptmodel.cli.main:app",
        ],
    },
    description="Prompt & model versioning on the cloud, built for developers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="weavel",
    url="https://github.com/weavel-ai/promptmodel",
    install_requires=[
        "pydantic",
        "peewee",
        "typer[all]",
        "cryptography",
        "pyyaml",
        "InquirerPy",
        "litellm",
        "python-dotenv",
        "websockets",
        "termcolor",
        "watchdog",
        "readerwriterlock",
    ],
    python_requires=">=3.7.1",
    keywords=[
        "weavel",
        "agent",
        "llm",
        "tools",
        "promptmodel",
        "llm agent",
        "prompt",
        "versioning",
        "eval",
        "evaluation",
    ],
)
