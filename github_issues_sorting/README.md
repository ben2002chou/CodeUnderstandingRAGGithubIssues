# Issues LLM Repository Script

1. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```


2. Run the script with the following command:

```bash
python3 issues_LLM_repo.py <repo-url> [max_issues] [--verbose] [--disable-wordnet]

Arguments

	•	<repo-url>: The URL of the GitHub repository (e.g., https://github.com/owner/repo).
	•	[max_issues]: (Optional) Maximum number of issues to fetch. Default is 100.
	•	--verbose: (Optional) Enable verbose output.
	•	--disable-wordnet: (Optional) Disable WordNet expansion of keywords. We find expansion does not help much, so it is reccomended to disable it.

Example

python3 issues_LLM_repo.py https://github.com/owner/repo 150 --verbose

Environment Variables

Create a .env file with your GitHub Personal Access Token:

GITHUB_TOKEN=your_github_personal_access_token

Output

The script outputs the top priority issues to the console and saves the results to <repo_name>_priority_issues.csv.

License

This project is licensed under the MIT License.

