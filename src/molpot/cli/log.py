import typer
from .main import app
from pathlib import Path


@app.command()
def log(
    log_dir: Path = typer.Argument(..., help="Path to the log directory."),
    port: int = typer.Option(6006, "--port", "-p", help="Port to run the server on."),
    backend: str = typer.Option("tensorboard", "--backend", "-b", help="Choose the backend to use.", ),
):
    """
    Process logs from the specified directory with an optional backend.
    """
    match backend:

        case "tensorboard":
            call_tensorboard(log_dir, port)

def call_tensorboard(log_dir: Path, port: int):
    import subprocess
    import sys

    try:
        subprocess.run(["tensorboard", "--logdir", str(log_dir), "--port", str(port)], check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
