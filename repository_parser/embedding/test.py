from openai import OpenAI
import pandas as pd
from generate_code_comment import (
    get_place_to_put,
    get_top_n_issue,
    fetch_issue_content,
    fetch_issue_comments,
    get_baseline_comment,
    generate_overall_code_comment,
    check_for_solution_in_comments,
    get_readme_description,
)
import ast
import os
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Function to evaluate a single example
def evaluate_example(
    function_body, baseline_comment, proposed_comment, evaluation_type
):
    """
    Use GPT-4 to evaluate the baseline and proposed comments based on updated criteria.
    """
    prompt = f"""
You are evaluating comments generated for a {evaluation_type}.

Function/README Content:
{function_body}

Baseline Comment:
{baseline_comment}

Proposed System Comment:
{proposed_comment}

Evaluate both comments on the following criteria:
1. Non-Obvious Insight: Does the comment provide useful information not obvious from the content?
2. Reduction of Setup Hassle: For README updates, does it make setup easier? For function comments, does it clarify the code purpose?
3. Helps Avoid Potential Issues: Does the comment highlight potential problems and provide guidance to prevent or address them?

Rate each on a scale of 1 to 5 and provide an overall ranking (1 = better, 2 = worse).

Output your response in this format:
Ratings:
Proposed System: Non-Obvious Insight: X, Reduction of Setup Hassle: X, Helps Avoid Potential Issues: X
Baseline: Non-Obvious Insight: X, Reduction of Setup Hassle: X, Helps Avoid Potential Issues: X
Ranking:
Proposed System: X
Baseline: X
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a developer reviewing generated comments.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during GPT-4 evaluation: {e}")
        return None


# Function to run the evaluation on multiple issues
def run_evaluation(csv_file, local_dir, num_issues=10):
    """
    Run evaluation for multiple issues, comparing the baseline and proposed comments.
    """
    results = {"function_related": [], "readme_related": []}
    df = pd.read_csv("./output/embedded.csv")
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    repo_des = get_readme_description(local_dir)

    for url, title in get_top_n_issue(csv_file, num_issues):
        # Fetch data
        content = fetch_issue_content(url)
        comments = fetch_issue_comments(url)

        # Get function body or fallback to README
        try:
            function_body, (file_path, start_line) = get_place_to_put(
                local_dir, url, title, df
            )
        except Exception as e:
            print(f"Error retrieving function body: {e}")
            function_body = ""

        # Determine additional context
        additional_context = function_body if function_body.strip() else repo_des

        # Determine the category of the issue
        if function_body.strip():
            issue_category = "function_related"
        elif "README" in file_path or not function_body.strip():
            issue_category = "readme_related"
        else:
            issue_category = "general"

        # Generate baseline comment
        baseline_comment = get_baseline_comment(function_body, repo_des)

        # Generate proposed system comment with context
        relevant_summaries = [
            check_for_solution_in_comments(comment) for comment in comments
        ]
        relevant_summaries = [
            summary for summary in relevant_summaries if summary
        ]  # Filter non-empty
        proposed_comment = generate_overall_code_comment(
            issue_title=title,
            issue_content=content,
            relevant_summaries="\n".join(relevant_summaries),
            additional_context=additional_context,
        )

        # Evaluate the comments
        evaluation_result = evaluate_example(
            function_body=additional_context,
            baseline_comment=baseline_comment,
            proposed_comment=proposed_comment,
            evaluation_type=issue_category.replace("_", " "),
        )

        # Append results based on category
        result_entry = {
            "issue_url": url,
            "issue_title": title,
            "issue_category": issue_category,
            "function_or_readme": additional_context,
            "baseline_comment": baseline_comment,
            "proposed_comment": proposed_comment,
            "evaluation_result": evaluation_result,
        }

        if issue_category == "function_related":
            results["function_related"].append(result_entry)
        elif issue_category == "readme_related":
            results["readme_related"].append(result_entry)

    # Save results to JSON files
    with open("evaluation_results_function.json", "w") as function_file:
        json.dump(results["function_related"], function_file, indent=4)
    with open("evaluation_results_readme.json", "w") as readme_file:
        json.dump(results["readme_related"], readme_file, indent=4)

    print(
        "Evaluation results saved to separate JSON files for function and README issues."
    )


# Example usage
if __name__ == "__main__":
    local_dir = r"/Users/Ben/Documents/GitHub/CodeUnderstandingRAGGithubIssues/repository_parser/embedding/output"  # Replace with the path to the cloned repo
    csv_file = r"/Users/Ben/Documents/GitHub/CodeUnderstandingRAGGithubIssues/github_issues_sorting/AutoGPT_priority_issues.csv"  # Replace with your actual file path

    # Run evaluation
    run_evaluation(csv_file=csv_file, local_dir=local_dir, num_issues=10)
