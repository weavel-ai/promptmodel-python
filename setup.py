"""
Prompt & model versioning on the cloud, built for developers.
"""
from setuptools import setup, find_namespace_packages

# Read README.md for the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="promptmodel",
    version="0.1.5",
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
        "httpx[http2]",
        "pydantic>=2.4.2",
        "peewee",
        "typer[all]",
        "cryptography",
        "pyyaml",
        "InquirerPy",
        "litellm>=1.7.1",
        # "litellm@git+https://github.com/weavel-ai/litellm.git@llms_add_clova_support",
        "python-dotenv",
        "websockets",
        "termcolor",
        "watchdog",
        "readerwriterlock",
        "nest-asyncio",
    ],
    python_requires=">=3.8.10",
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
        "collaborative",
    ],
)
