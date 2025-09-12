#!/usr/bin/env python3

import os
import sys

def generate_module_docs():

    for file in os.listdir('.'):
        if file.endswith('.rst') and file not in ['index.rst', 'conf.rst']:
            os.remove(file)
    
    print("Removed old .rst files")

    cmd = 'sphinx-apidoc -f -e -M -P -o . ..'
    print(f"Running: {cmd}")
    os.system(cmd)
    
    # Ensure modules.rst exists and has the right content
    with open('modules.rst', 'w') as f:
        f.write("API Reference\n")
        f.write("============\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 4\n\n")
        
        # Add all generated RST files that correspond to Python packages
        for file in sorted(os.listdir('.')):
            if file.endswith('.rst') and file not in ['index.rst', 'modules.rst', 'conf.rst']:
                module_name = file[:-4]  # Remove .rst extension
                f.write(f"   {module_name}\n")

if __name__ == '__main__':
    # Make sure we're in the docs directory
    current_dir = os.path.basename(os.getcwd())
    if current_dir != 'docs':
        print("This script should be run from the docs directory.")
        sys.exit(1)
    
    generate_module_docs()
    print("Run 'make html' to build the documentation.")
