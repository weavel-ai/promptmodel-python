"""Logger module"""

from typing import Any
import termcolor


def debug(msg: Any, *args):
    print(termcolor.colored("[DEBUG] " + str(msg) + str(*args), "light_yellow"))


def success(msg: Any, *args):
    print(termcolor.colored("[SUCCESS] " + str(msg) + str(*args), "green"))


def info(msg: Any, *args):
    print(termcolor.colored("[INFO] " + str(msg) + str(*args), "blue"))


def warning(msg: Any, *args):
    print(termcolor.colored("[WARNING] " + str(msg) + str(*args), "yellow"))


def error(msg: Any, *args):
    print(termcolor.colored("[Error] " + str(msg) + str(*args), "red"))
