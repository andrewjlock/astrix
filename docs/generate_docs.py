#!/usr/bin/env python3

import os
import sys
import subprocess


def generate_module_docs():
    # Clean up old RST files first (except the important ones)
    for file in os.listdir("."):
        if file.endswith(".rst") and file not in ["index.rst", "conf.rst"]:
            os.remove(file)

    # Run sphinx-apidoc with minimal options
    cmd = [
        "sphinx-apidoc",
        "-f",  # Force overwrite of existing files
        "--module-first",  # Module documentation first
        "-e",  # Documentation for each module on it's own page (?)
        "-o",  
        ".",
        "../src/astrix/",
        "../tests",
    ]
    subprocess.run(cmd, check=True)

    print("API documentation files generated. Run 'make html' to build the docs.")


if __name__ == "__main__":
    # Make sure we're in the docs directory
    if os.path.basename(os.getcwd()) != "docs":
        print("This script should be run from the docs directory.")
        sys.exit(1)

    generate_module_docs()
