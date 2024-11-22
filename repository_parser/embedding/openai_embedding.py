import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def get_repo_name_from_git_config(local_dir: str):
    """Retrieve the repository name from the .git/config file."""
    git_config_path = os.path.join(local_dir, ".git", "config")
    if os.path.exists(git_config_path):
        with open(git_config_path, "r") as file:
            for line in file:
                if "url = " in line:
                    url = line.split("url = ")[-1].strip()
                    repo_name = os.path.basename(url).replace(".git", "")
                    return repo_name
    return "UnknownRepository"


def get_readme_description(local_dir: str):
    """Generate a description based on the content of the README.md file."""
    readme_path = os.path.join(local_dir, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as file:
            readme_content = file.read()
        try:
            prompt = (
                "Provide a concise description of the following repository "
                f"based on its README:\n\n{readme_content}"
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            description = response.choices[0].message.content.strip()
            return description
        except Exception as e:
            print(f"Error generating description from README: {e}")
    return "Description unavailable."


# Set your local repository path
local_dir = "cloned_repo"  # Update to the actual path where your repository is cloned
print("current directory is: ", os.getcwd())
repo_name = get_repo_name_from_git_config(local_dir)
repo_description = get_readme_description(local_dir)

df = pd.read_csv("parsed.csv")  # Updated CSV file path
repository_background = (
    "Provide a repository background here."  # Added repository background
)
df["start_row"] = df["start_point"].str.extract(r"row=(\d+)")
df["end_row"] = df["end_point"].str.extract(r"row=(\d+)")
df["function_for_LLM"] = (
    f"This is a function from the {repo_name} repository. {repo_description} "
    "The file path is ./" + df["f_path"] + ". "
    "The function starts at line "
    + df["start_row"]
    + " and ends at line "
    + df["end_row"]
    + ". <function_body>"
    + df["function"]
    + "<function_body>"
)

# Ensure the output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Adding a progress bar to the embedding process
tqdm.pandas()


def embed_text(text):
    try:
        return get_embedding(text)
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None


df["embedding"] = df["function_for_LLM"].progress_apply(embed_text)
df.to_csv(os.path.join(output_dir, "embedded.csv"), index=False)
