import typer
import rich
from pathlib import Path

app = typer.Typer()

@app.command()
def potential(config: Path):
    
    ...