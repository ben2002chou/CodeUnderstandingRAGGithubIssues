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
    """Generate a short and concise description based on the content of the README.md file."""
    readme_path = os.path.join(local_dir, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as file:
            readme_content = file.read()

        # Preprocess README to extract key sections
        key_sections = []
        for section in [
            "# Introduction",
            "# Overview",
            "# Description",
            "# Usage",
            "# Features",
        ]:
            start = readme_content.lower().find(section.lower())
            if start != -1:
                end = readme_content.find(
                    "\n#", start + 1
                )  # Find the next section header
                key_sections.append(readme_content[start : end if end != -1 else None])

        # Truncate to the first X characters if necessary
        truncated_content = (
            "\n".join(key_sections) if key_sections else readme_content[:3000]
        )

        try:
            prompt = (
                "Provide a concise description of the following repository "
                "based on its README content:\n\n"
                f"{truncated_content}\n\n"
                "Keep the description short and focused."
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


if __name__ == "__main__":
    # Set your local repository path
    local_dir = "../cloned_repo_yt_dlp"  # Update to the actual path where your repository is cloned
    repo_name = get_repo_name_from_git_config(local_dir)
    repo_description = get_readme_description(local_dir)
    print("current path: ", os.getcwd())

    df = pd.read_csv("parsed_yt_dlp.csv")
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
    output_dir = "output_yt_dlp"
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
