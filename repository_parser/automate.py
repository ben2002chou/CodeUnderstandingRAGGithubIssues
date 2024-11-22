import subprocess
import os


def clone_repository(repo_url, clone_path):
    if os.path.exists(clone_path) and os.listdir(clone_path):
        print(
            f"Error: destination path '{clone_path}' already exists and is not an empty directory."
        )
        return False
    try:
        subprocess.run(["git", "clone", repo_url, clone_path], check=True)
        print(f"Repository cloned to {clone_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return False


def run_parser():
    try:
        subprocess.run(["python", "parser.py"], check=True)
        print("Parsing completed and CSV generated.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running parser.py: {e}")
        return False


def run_embedding():
    try:
        subprocess.run(["python", "embedding/openai_embedding.py"], check=True)
        print("Embedding completed and CSV generated.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running openai_embedding.py: {e}")
        return False


def find_code_block():
    try:
        subprocess.run(["python", "embedding/find_code_block.py"], check=True)
        print("Code block finding completed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running find_code_block.py: {e}")
        return False


if __name__ == "__main__":
    repo_url = "https://github.com/Significant-Gravitas/AutoGPT.git"
    clone_path = "./cloned_repo"

    if clone_repository(repo_url, clone_path):
        if run_parser():
            if run_embedding():
                find_code_block()
