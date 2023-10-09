import os
from typing import Any, Dict
from cryptography.fernet import Fernet
import secrets


def generate_api_key():
    """Generate a random token with 32 bytes"""
    token = secrets.token_hex(32)
    return token


def generate_crypto_key():
    """
    Generates a key and save it into a file
    """
    key = Fernet.generate_key()
    with open("key.key", "wb") as key_file:
        key_file.write(key)


def load_crypto_key():
    """
    Loads the key named `key.key` from the directory.
    Generates a new key if it does not exist.
    """
    if not os.path.exists("key.key"):
        generate_crypto_key()
    return open("key.key", "rb").read()


def encrypt_message(message: str) -> bytes:
    """
    Encrypts the message
    """
    key = load_crypto_key()
    f = Fernet(key)
    message = message.encode()
    encrypted_message = f.encrypt(message)
    return encrypted_message


def decrypt_message(encrypted_message: bytes) -> str:
    """
    Decrypts the message
    """
    key = load_crypto_key()
    f = Fernet(key)
    decrypted_message = f.decrypt(encrypted_message)
    return decrypted_message.decode()
