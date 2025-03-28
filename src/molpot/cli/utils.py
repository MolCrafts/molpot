import typer
from pathlib import Path
import logging

def get_version():
    return "0.1.0"

class SingletonCLI:

    _instances = None

    def __new__(cls, *args, **kwargs):
        if cls._instances is None:
            cls._instances = super(SingletonCLI, cls).__new__(cls)
        return cls._instances
    
    def __init__(self):
        self.cli = typer.Typer()
        self.logger = logging.getLogger("molpot")