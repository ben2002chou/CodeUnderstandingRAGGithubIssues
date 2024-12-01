from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
import requests
from bs4 import BeautifulSoup
from find_code_block import get_place_to_put
load_dotenv()
import ast
from openai_embedding import get_readme_description
from baseline import get_baseline_comment
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


def get_top_issue(csv_file):
    """Retrieve the top-priority issue from the CSV file."""
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="priority", ascending=False)  # Sort by priority
    top_issue = df.iloc[1]
    print(f"Top issue URL: {top_issue['url']}, Title: {top_issue['title']}")
    return top_issue["url"], top_issue["title"]

def get_top_n_issue(csv_file, n):
    """Retrieve the top-n issue from the CSV file."""
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="priority", ascending=False)  # Sort by priority
    top_issue = df.head(n)
    
    return zip(top_issue["url"], top_issue["title"])

def fetch_issue_content(url):
    """Fetch the content of a GitHub issue from the provided URL."""
    print(f"Fetching content from URL: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        issue_body = soup.find("td", {"class": "comment-body"})
        if issue_body:
            content = issue_body.get_text().strip()
            print(
                f"Fetched issue content: {content[:200]}..."
            )  # Print first 200 characters
            return content
    print("Unable to fetch issue content.")
    return "Unable to fetch issue content."


def fetch_issue_comments(url):
    """Fetch comments for a GitHub issue from the provided URL."""
    print(f"Fetching comments from URL: {url}")
    response = requests.get(url)
    comments = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        comment_bodies = soup.find_all("td", {"class": "comment-body"})
        for i, comment_body in enumerate(comment_bodies):
            comment_text = comment_body.get_text().strip()
            print(
                f"--- Comment {i+1} ---\n{comment_text[:300]}...\n"
            )  # Print first 300 characters of each comment
            comments.append(comment_text)
    else:
        print("Unable to fetch issue comments.")
    return comments


def check_for_solution_in_comments(comment):
    """Use OpenAI to check if a comment contains a solution or useful guidance."""
    prompt = f"""
    This is a GitHub issue comment related to a reported problem:
    {comment}

    Does this comment contain a suggested solution or useful guidance for the issue? If yes, summarize the solution in one or two sentences. If not, respond with 'No solution provided.'.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful summarizing AI."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
        )
        summary = response.choices[0].message.content.strip()
        if "No solution provided" not in summary:
            return summary
        else:
            return None  # Ignore non-solution comments
    except Exception as e:
        print(f"Error processing comment: {e}")
        return None


def generate_overall_code_comment(issue_title, issue_content, relevant_summaries):
    """Generate a final code comment summarizing the problem, available solutions, and guidance."""
    prompt = f"""
    Based on the following GitHub issue and its relevant comments, generate a code comment summarizing the problem, available solutions, and any guidance to prevent or address this issue in the future.

    Issue Title: {issue_title}
    Issue Content: {issue_content}

    Relevant Comments and Solutions:
    {relevant_summaries}

    Generate a concise and helpful code comment for developers.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful summarizing AI."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating overall comment: {e}")
        return "Error generating overall code comment."


# Example usage
if __name__ == "__main__":
    local_dir = r"D:\academic\A-LLMRec"  # Replace with the path to the cloned repo
    csv_file = r"D:\academic\CodeUnderstandingRAGGithubIssues\A-LLMRec_priority_issues.csv"  # Replace with your actual file path
    
    df = pd.read_csv("./output/embedded.csv")
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    repo_des = get_readme_description(local_dir)

    for url, title in get_top_n_issue(csv_file, 3):
        
        content = fetch_issue_content(url)
        comments = fetch_issue_comments(url)
        function_body, (file_path, start_line) = get_place_to_put(url, title, df)
        #baseline
        baseline_result = get_baseline_comment(function_body, repo_des)

        #our method
        # Check for solutions in each comment
        relevant_summaries = []
        for comment in comments:
            summary = check_for_solution_in_comments(comment)
            if summary:  # Only consider comments that contain a solution
                relevant_summaries.append(summary)

        overall_comment = generate_overall_code_comment(
            title, content, "\n".join(relevant_summaries)
        )
        print("\n--- Overall Code Comment ---\n")
        print(overall_comment)
