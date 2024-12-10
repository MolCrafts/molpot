import typer
import rich
from pathlib import Path
from typing_extensions import Annotated

app = typer.Typer()

@app.command()
def train(
    script: Annotated[Path, typer.Option(help="Path to the script to train the model")] = None,
    config: Annotated[Path, typer.Option(help="Path to the config file")] = None,
):
    assert not (script and config), "You can only provide one of script or config"

    if script:
        train_with_script(script)

@app.command()
def train_with_script(script: Annotated[Path, typer.Argument(help="Path to the script to train the model")]):
    print(script)

