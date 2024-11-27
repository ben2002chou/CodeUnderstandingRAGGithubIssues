import os
import subprocess
import pandas as pd
from tree_sitter import Language, Parser
import tree_sitter_python
from repository_parser.repository_parser.python_parser import *

PYTHON_LANGUAGE = Language(tree_sitter_python.language())


def clone_repository(repo_url: str, local_dir: str):
    try:
        subprocess.run(["git", "clone", repo_url, local_dir], check=True)
        print(f"Repository cloned to {local_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")


def traverse_and_parse(folder_path: str):
    parser = Parser(PYTHON_LANGUAGE)
    all_definitions = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()
                    code_bytes = code.encode("utf-8")
                    tree = parser.parse(code_bytes)
                    definitions = PythonParser.get_definition(tree, code_bytes)
                    for item in definitions:
                        # Add file path information
                        relative_path = os.path.relpath(file_path, folder_path)
                        item["f_path"] = relative_path.replace("\\", "/")
                        all_definitions.append(item)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    return all_definitions


if __name__ == "__main__":
    repo_url = "https://github.com/Significant-Gravitas/AutoGPT.git"  # Replace with your repo link
    local_dir = "cloned_repo"  # Directory where the repo will be cloned

    # Clone the repository
    clone_repository(repo_url, local_dir)

    # Parse the cloned repository
    definitions = traverse_and_parse(local_dir)

    # Save the definitions to a CSV file
    prj_name = os.path.basename(repo_url).replace(".git", "")
    df = pd.DataFrame(definitions)
    output_csv_path = f"parsed.csv"
    df.to_csv(output_csv_path, index=False)
    print(df.head())
