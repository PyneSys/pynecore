import os
import sys

from pathlib import Path

from typer import Typer, Argument, Exit, secho

from ..app import app, app_state

__all__ = []

app_debug = Typer(help="Debug tools")
app.add_typer(app_debug, name="debug")


@app_debug.command()
def ast(
        script: Path = Argument(
            ...,
            help="Script file to transform (.py or .pine)",
        ),
):
    """
    Show AST-transformed code without running the script.
    """
    # Expand script path
    if len(script.parts) == 1:
        script = app_state.scripts_dir / script

    # If no suffix, try .py first (we want the compiled Python, not Pine)
    if script.suffix == "":
        py_path = script.with_suffix(".py")
        if py_path.exists():
            script = py_path
        else:
            script = script.with_suffix(".pine")

    # Handle .pine -> .py fallback
    if script.suffix == ".pine" and not script.exists():
        script = script.with_suffix(".py")

    if not script.exists():
        secho(f"Script file '{script}' not found!", fg="red", err=True)
        raise Exit(1)

    if script.suffix == ".pine":
        secho("Pine Script files must be compiled first: pyne compile <script>", fg="red", err=True)
        raise Exit(1)

    # Set debug env var so the import hook prints the transformed code
    os.environ['PYNE_AST_DEBUG'] = '1'

    try:
        from pynecore.core.script_runner import import_script
        import_script(script)
    except Exception as e:
        secho(f"Error: {e}", fg="red", err=True)
        raise Exit(1)
    finally:
        os.environ.pop('PYNE_AST_DEBUG', None)
        # Clean up the imported module from sys.modules
        module_name = script.stem
        sys.modules.pop(module_name, None)
