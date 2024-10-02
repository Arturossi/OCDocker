import os
import ast
import graphviz

def find_python_files(folder_path):
    """
    Find all Python files in the given folder path and its subdirectories.
    
    :param folder_path: Path to the folder containing Python files.
    :return: A list of paths to Python files.
    """
    python_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

def parse_imports(file_path, module_names, project_prefix, verbose=False, ignore_init_files=True, ignore_deprecated=True):
    """
    Parse the imports of a Python file to detect dependencies on other modules within the project.
    
    :param file_path: Path to the Python file.
    :param module_names: Dictionary mapping module names to their file paths.
    :param project_prefix: The root module namespace for the project (e.g., 'OCDocker').
    :param verbose: Boolean flag for printing detailed processing information.
    :param ignore_init_files: Boolean flag to ignore __init__.py files (default: True).
    :param ignore_deprecated: Boolean flag to ignore imports within Deprecated or deprecated directories (default: True).
    :return: A list of imported module names within the project.
    """
    if ignore_init_files and os.path.basename(file_path) == '__init__.py':
        if verbose:
            print(f"Ignoring __init__.py file: {file_path}")
        return []

    if ignore_deprecated and ('Deprecated' in file_path or 'deprecated' in file_path):
        if verbose:
            print(f"Ignoring deprecated file: {file_path}")
        return []

    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)

    imports = []
    for node in ast.walk(tree):
        # Handle absolute imports and aliased imports (e.g., "import OCDocker.Toolbox.Conversion as occonv")
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name  # Capture full module name
                # Check if it's a project-specific import by prefix
                if module_name.startswith(project_prefix):
                    imports.append(module_name)
                    if verbose:
                        print(f"Detected project-specific import: {module_name} in {file_path}")
                else:
                    if verbose:
                        print(f"Ignoring external import: {module_name} in {file_path}")
        # Handle "from ... import ..." (e.g., "from OCDocker.Toolbox import Conversion")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module  # Capture full module name
                # Check if it's a project-specific import by prefix
                if module_name.startswith(project_prefix):
                    imports.append(module_name)
                    if verbose:
                        print(f"Detected project-specific import: {module_name} in {file_path}")
                else:
                    if verbose:
                        print(f"Ignoring external import: {module_name} in {file_path}")
            elif node.level > 0:
                # Handle relative imports like "from .module import ..."
                current_module = os.path.splitext(os.path.basename(file_path))[0]
                rel_module = resolve_relative_import(node.level, current_module, module_names)
                if rel_module and rel_module in module_names:
                    imports.append(rel_module)
                    if verbose:
                        print(f"Resolved relative import: {rel_module} in {file_path}")
                else:
                    if verbose:
                        print(f"Could not resolve relative import: level={node.level}, module={current_module} in {file_path}")

    return imports

def resolve_relative_import(level, current_module, module_names):
    """
    Resolve the full module name from a relative import.

    :param level: The level of the relative import (e.g., `from .. import ...` is level 2).
    :param current_module: The current module name.
    :param module_names: A dictionary of module names to file paths.
    :return: The resolved module name.
    """
    parts = current_module.split(".")
    if len(parts) >= level:
        resolved_module = ".".join(parts[:-level])
        if resolved_module in module_names:
            return resolved_module
    return None

def map_dependencies(folder_path, project_prefix, verbose=False):
    """
    Map dependencies between Python files in the given folder by parsing their imports.
    
    :param folder_path: Path to the folder containing Python files.
    :param project_prefix: The root module namespace for the project (e.g., 'OCDocker').
    :param verbose: Boolean flag for printing detailed processing information.
    :return: A dictionary where the keys are Python file names, and the values are lists of dependencies.
    """
    python_files = find_python_files(folder_path)
    dependencies = {}

    # Build a dictionary of file names (without .py extension) -> full paths
    module_names = {}
    for file in python_files:
        module_name = os.path.relpath(file, folder_path).replace("/", ".").replace("\\", ".").replace(".py", "")
        module_names[module_name] = file

    if verbose:
        print(f"Module names detected: {module_names.keys()}")  # Debugging statement to show detected modules

    # Map dependencies based on imports
    for module_name, file_path in module_names.items():
        imports = parse_imports(file_path, module_names, project_prefix, verbose=verbose)
        dependencies[module_name] = imports
        if verbose:
            print(f"Module '{module_name}' imports: {imports}")  # Debugging statement for detected imports

    return dependencies

def generate_dependency_graph(dependencies, output_file, format='png', engine='dot'):
    """
    Generate a Graphviz DOT file and render a dependency graph from the dependencies map.
    
    :param dependencies: A dictionary mapping Python files to their dependencies.
    :param output_file: The name of the output image file (without extension).
    :param format: The format of the output file (default is 'png').
    :param engine: The Graphviz engine to use (default is 'dot').
    """
    dot = graphviz.Digraph(format=format, engine=engine)

    dot.attr(rankdir="LR") 

    for module, imports in dependencies.items():
        for imp in imports:
            dot.edge(module, imp)  # Create an edge between the module and its dependency
    
    # Save the DOT file and render the image
    dot.render(output_file, view=False)
    print(f"Dependency graph saved as {output_file}.{format} using {engine} engine")

# Path to the project folder containing Python files
project_folder = './OCDocker'  # Change this to your folder path
project_prefix = 'OCDocker'  # The root module namespace for your project

# Map the dependencies
dependencies = map_dependencies(project_folder, project_prefix)

# Generate and save the dependency graph with format and engine options
generate_dependency_graph(dependencies, 'dependencies_graph', format='png', engine='twopi')

# Available Graphviz Engines:

# dot: A hierarchical layout engine (default).
# neato: A spring model layout engine.
# fdp: A force-directed layout.
# sfdp: A scalable version of fdp for large graphs.
# twopi: A radial layout engine.
# circo: A circular layout engine.
# osage: Draws clustered graphs.
# patchwork: Draws map of clustered graph using a squarified treemap layout.

for engine in ['dot', 'neato', 'fdp', 'sfdp', 'twopi', 'circo', 'osage', 'patchwork']:
    generate_dependency_graph(dependencies, f'dependencies_graph_{engine}', format='png', engine=engine)

# Available Formats:

# png: A common raster image format.
# svg: A vector image format.
# pdf: A PDF file format.
# jpg, jpeg: Raster image formats.