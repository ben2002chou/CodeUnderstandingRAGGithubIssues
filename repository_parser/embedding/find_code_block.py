from dotenv import load_dotenv
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from scipy import spatial  # for calculating vector similarities for search
import ast

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


def get_embedding(text, model="text-embedding-3-small"):
    """Generate an embedding for the provided text using OpenAI API."""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def get_top_issue(csv_file):
    """Retrieve the top-priority issue from the CSV file."""
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="priority", ascending=False)  # Sort by priority
    top_issue = df.iloc[0]  # 0
    print(f"Top issue URL: {top_issue['url']}, Title: {top_issue['title']}")
    return top_issue["url"], top_issue["title"]


def fetch_issue_content_with_comments(
    url, max_comments=3, max_token_limit=8192, comment_trunc_limit=1000
):
    """Fetch the issue content and the first few comments from a GitHub issue, with optional truncation."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        # Fetch the main issue content
        issue_body = soup.find("td", {"class": "comment-body"})
        issue_content = (
            issue_body.get_text().strip()
            if issue_body
            else "Unable to fetch issue content."
        )
        combined_content = f"Issue Content:\n{issue_content}\n\n"

        # Fetch the first few comments
        comments = soup.find_all("td", {"class": "comment-body"})
        num_comments = min(
            len(comments) - 1, max_comments
        )  # Exclude the main issue body
        for i in range(1, num_comments + 1):  # Start from the first comment
            comment_text = comments[i].get_text().strip()
            # Truncate comment if it exceeds the limit
            if len(comment_text) > comment_trunc_limit:
                comment_text = comment_text[:comment_trunc_limit] + "..."
            combined_content += f"--- Comment {i} ---\n{comment_text}\n\n"

        # Truncate the combined content if it exceeds the token limit
        if len(combined_content) > max_token_limit:
            combined_content = combined_content[:max_token_limit] + "..."

        return combined_content
    return "Unable to fetch issue content."


def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
):
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["function"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


# Example usage
local_dir = "../cloned_repo"  # Replace with the path to the cloned repo if needed
csv_file = "/Users/Ben/Documents/GitHub/CodeUnderstandingRAGGithubIssues/github_issues_sorting/AutoGPT_priority_issues.csv"  # Replace with your actual file path
url, title = get_top_issue(csv_file)
content = fetch_issue_content_with_comments(url)

print("Fetched content for embedding:")
print(content[:500])  # Print the first 500 characters to check the content

# Generate embedding for the fetched content
issue_embedding = get_embedding(content)
print("Generated issue embedding.")

# Load embeddings from a CSV file
df = pd.read_csv("./output/embedded.csv")
df["embedding"] = df["embedding"].apply(ast.literal_eval)

# Find the most related code blocks based on the issue content embedding
strings, relatednesses = strings_ranked_by_relatedness(content, df, top_n=5)
print("\nTop related code blocks:")
for string, relatedness in zip(strings, relatednesses):
    print(f"{relatedness=:.3f}")
    print(string)
