import pkgutil
import os
from graphviz import Digraph


def print_package_structure(package_path, package_name, dot=None):
    if dot is None:
        dot = Digraph()

    # Add the main package node
    dot.node(package_name, package_name)

    # Iterate through the modules in the package
    for importer, modname, ispkg in pkgutil.iter_modules([package_path]):
        # Check for directories to ignore
        if 'Deprecated' in modname or 'deprecated' in modname:
            continue
        
        # Ignore __init__ and __main__ modules
        if modname in ['__init__', '__main__']:
            continue

        # Add subpackage or module node
        dot.node(modname, modname)
        dot.edge(package_name, modname)  # Connect to the main package

        # If it's a subpackage, recursively print its structure
        if ispkg:
            subpackage_path = os.path.join(package_path, modname)
            print_package_structure(subpackage_path, modname, dot)

    return dot


def create_local_package_structure_diagram(package_dir, output_format='png', engine='dot', rankdir='TB'):
    # Ensure the specified path is a directory
    if not os.path.isdir(package_dir):
        raise ValueError(f"{package_dir} is not a valid directory.")

    package_name = os.path.basename(package_dir)
    dot = print_package_structure(package_dir, package_name)

    # Set the engine and rankdir
    dot.attr(rankdir = rankdir)  # Set the graph direction
    dot.engine = engine

    # Save and render the diagram
    dot.render(f'{package_name}_structure_{engine}_{rankdir}', format=output_format, cleanup=True)

# Example usage
package_directory = '/data/hd4tb/OCDocker/OCDocker'  # Replace with your package directory
output_driver = 'png'  # Specify your desired output format (e.g., 'png', 'pdf', 'svg')
layout_engine = 'dot'  # Specify the layout engine (e.g., 'dot', 'sfdp', 'neato', etc.)
rank_direction = 'TB'  # Specify rankdir (e.g., 'TB' for top-to-bottom, 'LR' for left-to-right)

for rank_direction in ['TB', 'LR']:
    for engine in ['dot', 'sfdp', 'neato', 'fdp', 'circo', 'twopi', 'osage', 'patchwork']:
        layout_engine = engine
        create_local_package_structure_diagram(package_directory, output_driver, layout_engine, rank_direction)
