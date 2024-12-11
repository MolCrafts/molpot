import typer
import rich
from pathlib import Path
from typing_extensions import Annotated
from typing import Any
import runpy
import sys

app = typer.Typer()

@app.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def script(ctx: typer.Context, script: Annotated[Path, typer.Argument(help="Path to the script to train the model")]):
    original_argv = sys.argv[:]  # Backup original sys.argv
    try:
        sys.argv = [script] + ctx.args  # Inject arguments into sys.argv
        runpy.run_path(script, run_name="__main__")  # Run the target script
    except Exception as e:
        raise e
    finally:
        sys.argv = original_argv  # Restore original sys.argv