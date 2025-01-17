import typer

app = typer.Typer()

@app.callback(invoke_without_command=True)
def default_welcome(ctx: typer.Context):
    version = "0.1.0"
    if ctx.invoked_subcommand is None:
        typer.echo(r"""
                    __            __ 
   ____ ___  ____  / /___  ____  / /_
  / __ `__ \/ __ \/ / __ \/ __ \/ __/
 / / / / / / /_/ / / /_/ / /_/ / /_  
/_/ /_/ /_/\____/_/ .___/\____/\__/  
                 /_/                 
""")  # generator: https://patorjk.com/software/taag/#p=display&f=Slant&t=molpot
        typer.echo(f"Welcome to MolPot {version}")
        typer.echo()
        typer.echo(ctx.get_help())

if __name__ == "__main__":
    app()