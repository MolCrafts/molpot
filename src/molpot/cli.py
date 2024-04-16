import typer
import importlib
from pathlib import Path

app = typer.Typer()

@app.command()
def run(script_path: str, entry_point: str="main"):
    spath = Path(script_path)
    script_name = spath.stem
    Path.cwd().joinpath(script_name).mkdir(exist_ok=True)
    typer.echo(f"Script loaded {script_name} from {spath}")
    import sys
    sys.path.append(str(spath.parent))
    script = importlib.import_module(script_name)
    entry_point = getattr(script, entry_point)
    typer.echo(f"Running {entry_point.__name__}")
    result = entry_point()
    typer.echo(result)

@app.command()
def welcome():
    version = "0.0.1"
    typer.echo(f"Welcome to MolPot {version}")
