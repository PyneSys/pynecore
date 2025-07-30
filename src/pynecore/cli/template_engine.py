from pathlib import Path
from typing import Dict, Any
import re
from datetime import datetime


class TemplateEngine:
    """Simple template engine for plugin generation"""
    
    def __init__(self):
        self.variables = {}
    
    def set_variables(self, variables: Dict[str, Any]):
        """Set template variables"""
        self.variables.update(variables)
    
    def render(self, template_content: str) -> str:
        """Render template with variables"""
        result = template_content
        
        # Replace variables in format {{variable_name}}
        for key, value in self.variables.items():
            pattern = rf"{{{{\s*{key}\s*}}}}"
            result = re.sub(pattern, str(value), result)
        
        return result
    
    def render_file(self, template_path: Path, output_path: Path):
        """Render template file to output file"""
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        rendered_content = self.render(template_content)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rendered_content)


def get_plugin_templates_dir() -> Path:
    """Get the plugin templates directory"""
    return Path(__file__).parent / "templates" / "plugins"


def get_default_template_variables(plugin_name: str, plugin_type: str) -> Dict[str, Any]:
    """Get default template variables for plugin generation"""
    # Convert plugin name to various formats
    plugin_name_snake = plugin_name.lower().replace('-', '_').replace(' ', '_')
    plugin_name_pascal = ''.join(word.capitalize() for word in plugin_name_snake.split('_'))
    plugin_name_kebab = plugin_name_snake.replace('_', '-')
    
    # Get PyneCore version from main pyproject.toml
    pynecore_version = _get_pynecore_version()
    
    return {
        'plugin_name': plugin_name,
        'plugin_name_snake': plugin_name_snake,
        'plugin_name_pascal': plugin_name_pascal,
        'plugin_name_kebab': plugin_name_kebab,
        'plugin_type': plugin_type,
        'current_year': datetime.now().year,
        'current_date': datetime.now().strftime('%Y-%m-%d'),
        'pynecore_version': pynecore_version,
    }


def _get_pynecore_version() -> str:
    """Get PyneCore version from PyPI"""
    try:
        import subprocess
        import json
        
        # Use curl to fetch version from PyPI API (avoids SSL issues)
        url = "https://pypi.org/pypi/pynesys-pynecore/json"
        result = subprocess.run(
            ["curl", "-s", "--max-time", "10", url],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            version = data.get('info', {}).get('version', '0.1.0')
            return f">={version}"
    except Exception:
        pass
    
    # Fallback on any error - try to get from local pyproject.toml
    try:
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            import tomllib
            with open(pyproject_path, 'rb') as f:
                data = tomllib.load(f)
                version = data.get('project', {}).get('version', '0.1.0')
                return f">={version}"
    except Exception:
        pass
    
    # Final fallback
    return ">=0.1.0"