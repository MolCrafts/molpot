import typer

from .train import app as train

app = typer.Typer()
app.add_typer(train, name="train")

@app.callback(invoke_without_command=True)
def default_welcome(ctx: typer.Context):
    version = "0.1.0"
    typer.echo("# -------------")
    if ctx.invoked_subcommand is None:
        typer.echo(f"Welcome to MolPot {version}")
        typer.echo()
        typer.echo(ctx.get_help())
