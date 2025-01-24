import typer
from pathlib import Path
import runpy
from .main import app
import importlib

@app.command()
def train(
    script: Path = typer.Argument(..., help="Path to the training script."),
):
    """
    Train a model using the specified script.
    """
    ...