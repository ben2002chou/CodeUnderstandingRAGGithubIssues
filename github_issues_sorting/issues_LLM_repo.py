#!/usr/bin/env python3

"""
GitHub Issues Prioritization Script with Deduplication
------------------------------------------------------

This script fetches issues from a specified GitHub repository, calculates priority scores based on various factors,
and outputs the top priority issues. It includes caching and deduplication logic to avoid processing the same issues
multiple times.

Usage:
    python3 issues_LLM_repo.py <repo-url> [max_issues] [--verbose] [--disable-wordnet]

Example:
    python3 issues_LLM_repo.py https://github.com/owner/repo 150 --verbose
"""

import requests
import json
import sys
import re
import os
import time
import argparse
import logging
import csv
from typing import Optional, List, Dict, Set, Tuple
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def run_query(auth_token: str, query: str, variables: Dict, max_retries: int = 3) -> Dict:
    """
    Executes a GraphQL query to the GitHub API with retry logic and rate limit handling.
    """
    url = "https://api.github.com/graphql"
    headers = {"Authorization": f"Bearer {auth_token}"}

    for attempt in range(1, max_retries + 1):
        response = requests.post(
            url, json={"query": query, "variables": variables}, headers=headers
        )

        if response.status_code == 200:
            # Handle rate limit
            remaining = int(response.headers.get('X-RateLimit-Remaining', 1))
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            if remaining == 0:
                sleep_time = max(reset_time - time.time(), 0)
                logging.warning(f"Rate limit reached. Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
            return response.json()

        elif response.status_code in [502, 503, 504]:
            # Retry on server errors
            wait_time = 2 ** attempt
            logging.warning(f"Server error {response.status_code}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        elif response.status_code == 403 and 'rate limit' in response.text.lower():
            # Rate limit exceeded
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
            sleep_time = max(reset_time - time.time(), 0)
            logging.warning(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
        else:
            # For other errors, raise an exception
            raise Exception(
                f"Query failed with status code {response.status_code}: {response.text}"
            )

    raise Exception("Max retries exceeded")

def extract_owner_and_repo_from_url(repo_url: str) -> Tuple[str, str]:
    """
    Extracts the owner and repository name from a GitHub repository URL.
    """
    # Remove any trailing '.git' or '/'
    repo_url = repo_url.rstrip("/").rstrip(".git")
    split_url = repo_url.split("/")
    if len(split_url) < 5:
        raise ValueError(
            "Invalid GitHub repository URL format. Expected format: https://github.com/<owner>/<repository-name>"
        )
    owner = split_url[3]
    repo_name = split_url[4]
    return owner, repo_name

def is_bot_user(user_login: str) -> bool:
    """
    Determines if a user is a bot based on their login name.
    """
    if not user_login:
        return False
    bot_indicators = ["[bot]", "-bot", "bot"]
    user_login_lower = user_login.lower()
    return any(
        user_login_lower.endswith(indicator) or indicator in user_login_lower
        for indicator in bot_indicators
    )

def get_contributors(auth_token: str, owner: str, repo_name: str) -> Set[str]:
    """
    Retrieves a set of contributors to the specified repository, excluding bots.
    """
    cache_file = f"{repo_name}_contributors.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            all_contributors = json.load(f)
        logging.info(f"Loaded contributors from cache ({cache_file}).")
    else:
        query = """
        query ($owner: String!, $name: String!, $cursor: String) {
          repository(owner: $owner, name: $name) {
            mentionableUsers(first: 50, after: $cursor) {
              pageInfo {
                endCursor
                hasNextPage
              }
              nodes {
                login
              }
            }
          }
        }
        """
        variables = {
            "owner": owner,
            "name": repo_name,
            "cursor": None,
        }
        all_contributors = []

        while True:
            result = run_query(auth_token, query, variables)

            # Check for errors
            if "errors" in result:
                logging.error(f"Errors in get_contributors: {result['errors']}")
                return set()

            repository = result.get("data", {}).get("repository")
            if repository is None:
                logging.error(f"Repository {owner}/{repo_name} not found or inaccessible.")
                return set()

            mentionable_users = repository.get("mentionableUsers")
            if mentionable_users:
                users = mentionable_users["nodes"]
                # Exclude bots using is_bot_user function
                contributors = [
                    user["login"] for user in users if not is_bot_user(user["login"])
                ]
                all_contributors.extend(contributors)

                page_info = mentionable_users["pageInfo"]
                if page_info["hasNextPage"]:
                    variables["cursor"] = page_info["endCursor"]
                else:
                    break
            else:
                break
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(all_contributors, f)
        logging.info(f"Saved contributors to cache ({cache_file}).")

    return set(all_contributors)

def get_issues(auth_token: str, owner: str, name: str, states: Optional[List[str]] = None, max_issues: Optional[int] = None) -> Optional[List[Dict]]:
    """
    Fetches issues from the specified repository using pagination.
    """
    if states is None:
        states = ['OPEN', 'CLOSED']  # Fetch both open and closed issues

    query = """
    query ($owner: String!, $name: String!, $cursor: String, $issues_per_page: Int!, $states: [IssueState!]) {
      repository(owner: $owner, name: $name) {
        issues(first: $issues_per_page, after: $cursor, states: $states, orderBy: {field: CREATED_AT, direction: DESC}) {
          pageInfo {
            endCursor
            hasNextPage
          }
          nodes {
            url
            number
            title
            state
            createdAt
            updatedAt
            author {
              login
            }
            labels(first: 5) {
              nodes {
                name
                isDefault
              }
            }
            comments(first: 5) {
              nodes {
                author {
                  login
                }
                bodyText
              }
            }
            reactions(first: 5) {
              totalCount
            }
          }
        }
      }
    }
    """
    variables = {
        "owner": owner,
        "name": name,
        "cursor": None,
        "issues_per_page": 50,
        "states": states,
    }

    all_issues = []
    total_fetched = 0  # Counter for the number of issues fetched

    while True:
        result = run_query(auth_token, query, variables)

        # Check for errors
        if "errors" in result:
            logging.error(f"Errors in get_issues: {result['errors']}")
            return None

        repository = result.get("data", {}).get("repository")
        if repository is None:
            logging.error(f"Repository {owner}/{name} not found or inaccessible.")
            return None

        issues_data = repository.get("issues")
        if issues_data:
            issues_nodes = issues_data["nodes"]
            all_issues.extend(issues_nodes)
            total_fetched += len(issues_nodes)

            # Check if we've reached the maximum number of issues
            if max_issues is not None and total_fetched >= max_issues:
                # Trim the list to the maximum number of issues
                all_issues = all_issues[:max_issues]
                break

            page_info = issues_data["pageInfo"]
            if page_info["hasNextPage"]:
                variables["cursor"] = page_info["endCursor"]
            else:
                break
        else:
            break

    return all_issues

def expand_keywords_with_synonyms(keywords: List[str], use_wordnet: bool = True) -> List[str]:
    """
    Expands a list of keywords with their synonyms using WordNet and lemmatization.

    Args:
        keywords (List[str]): The initial list of keywords.
        use_wordnet (bool): Whether to use WordNet expansion.

    Returns:
        List[str]: A list of expanded keywords.
    """
    # Lemmatize initial keywords
    lemmatized_keywords = {lemmatizer.lemmatize(keyword.lower()) for keyword in keywords}

    if not use_wordnet:
        return list(lemmatized_keywords)

    synonyms = set(lemmatized_keywords)
    for keyword in lemmatized_keywords:
        for syn in wn.synsets(keyword):
            for lemma in syn.lemmas():
                synonym = lemmatizer.lemmatize(lemma.name().replace('_', ' ').lower())
                synonyms.add(synonym)
    # Filter out irrelevant terms
    irrelevant_terms = {'problem', 'job', 'servant'}  # Add any irrelevant terms here
    synonyms -= irrelevant_terms
    return list(synonyms)

# Pre-compile regular expressions
code_block_patterns = [re.compile(r"```"), re.compile(r"`")]

explanatory_phrases = [re.compile(re.escape(phrase)) for phrase in [
    "this function",
    "here we",
    "in order to",
    "the code does",
    "because",
    "for example",
    "we use",
    "this code",
    "explained",
    "as shown",
    "the purpose of this code",
    "we implemented",
    "this algorithm",
    "to achieve this",
    "as a result",
    "in this example",
    "this allows us to",
    "it works by",
    "first, we",
    "then, we",
    "finally, we",
    "our approach",
    "we can see that",
    "this demonstrates",
    "we need to",
    "this method",
    "as demonstrated",
    "the reason is",
    "this is because",
    "the following code",
    "here's how",
    "step by step",
    "consider the following",
    "note that",
    "pay attention to",
    "as you can see",
    "the logic behind",
    "this snippet",
    "the implementation of",
    "to work",
    "should work",
    "works",
    "worked",
    "working",
]]

def is_question_or_documentation(issue: Dict, expanded_keywords: List[str]) -> bool:
    """
    Determines if an issue is related to a question or documentation.
    """
    # Define labels that indicate question or documentation
    question_labels = {"question", "documentation", "docs", "help wanted"}
    issue_labels = {label["name"].lower() for label in issue.get("labels", {}).get("nodes", [])}

    # Check if any of the issue labels match the desired labels
    if issue_labels.intersection(question_labels):
        return True

    # Lemmatize the title words
    title = issue.get("title", "").lower()
    title_words = [lemmatizer.lemmatize(word) for word in word_tokenize(title)]

    # Since expanded_keywords are already lemmatized, we can proceed to match
    if set(expanded_keywords).intersection(title_words):
        return True

    return False

def should_exclude_issue(issue: Dict) -> bool:
    """
    Determines if an issue should be excluded based on certain labels.
    """
    # Define labels that indicate issues to exclude
    exclude_labels = {
        "bug",
        "confirmed bug",
        "type: bug",
        "kind/bug",
        "duplicate",
        "status: duplicate",
        "type: duplicate",
        "invalid",
        "type: invalid",
        "wontfix",
        "won't fix",
        "status: wontfix",
        "spam",
    }
    issue_labels = {label["name"].lower() for label in issue.get("labels", {}).get("nodes", [])}

    intersection = issue_labels.intersection(exclude_labels)
    if intersection:
        logging.debug(f"Excluding issue {issue['url']} due to labels: {intersection}")
        return True

    return False

def has_explanatory_comment(issue_comments: List[Dict]) -> bool:
    """
    Checks if any of the issue comments contain explanatory code snippets.
    """
    for comment in issue_comments:
        if not comment or not comment.get("author"):
            continue  # Skip if comment or author is missing
        author_login = comment["author"].get("login")
        if is_bot_user(author_login):
            continue  # Skip comments by bots
        body_text = comment.get("bodyText", "").lower()
        # Lemmatize the body text
        lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(body_text)])
        # Check for code blocks
        has_code_block = any(pattern.search(body_text) for pattern in code_block_patterns)
        # Check for explanatory phrases
        has_explanatory_phrase = any(
            pattern.search(lemmatized_text) for pattern in explanatory_phrases
        )
        if has_code_block and has_explanatory_phrase:
            return True
    return False

def calculate_priority(issue: Dict, contributors: Set[str], repo_owner: str, expanded_keywords: List[str]) -> Optional[float]:
    """
    Calculates a priority score for an issue based on various factors.
    """
    score = 0

    # Skip the issue if it should be excluded
    if should_exclude_issue(issue):
        return None  # Indicate that this issue should be skipped

    # Extract author login safely
    author = issue.get("author")
    if author and author.get("login"):
        author_login = author["login"]
    else:
        author_login = None

    # Exclude the issue if the author is a bot
    if author_login and is_bot_user(author_login):
        return None  # Skip issues authored by bots

    # Whether or not used a default label
    has_default_label = any(
        label.get("isDefault", False)
        for label in issue.get("labels", {}).get("nodes", [])
    )
    if has_default_label:
        score += 2  # Adjusted weight

    # Title length
    title_length = len(issue.get("title", ""))
    score += min(title_length / 50, 2)  # Cap at 2 points

    # If repository owner or contributor has responded (exclude bots)
    commenters = {
        comment.get("author", {}).get("login")
        for comment in issue.get("comments", {}).get("nodes", [])
        if comment.get("author")
        and comment["author"].get("login")
        and not is_bot_user(comment["author"]["login"])
    }
    commenters.discard(None)  # Remove None values
    if repo_owner in commenters:
        score += 5  # Repository owner has commented
    elif commenters.intersection(contributors):
        score += 3  # A contributor has commented

    # If issue is closed
    if issue.get("state") == "CLOSED":
        score += 1

    # Who reported the issue? If contributor or owner
    if author_login:
        if author_login == repo_owner:
            score += 5
        elif author_login in contributors:
            score += 3
        else:
            score += 1

    # Number of reactions
    reactions_count = issue.get("reactions", {}).get("totalCount", 0)
    score += min(reactions_count, 5)  # Cap at 5 points

    # Additional points if issue is a question or documentation
    if is_question_or_documentation(issue, expanded_keywords):
        score += 7

    # Additional points if issue has an explanatory comment
    if has_explanatory_comment(issue.get("comments", {}).get("nodes", [])):
        score += 15

    return score

def update_cached_issues(cached_issues: List[Dict], new_issues: List[Dict]) -> List[Dict]:
    """
    Updates the cached issues with new data from fetched issues.

    Args:
        cached_issues (List[Dict]): The list of cached issues.
        new_issues (List[Dict]): The list of newly fetched issues.

    Returns:
        List[Dict]: The updated list of cached issues.
    """
    cached_issues_dict = {issue['number']: issue for issue in cached_issues}
    for issue in new_issues:
        cached_issues_dict[issue['number']] = issue
    return list(cached_issues_dict.values())

if __name__ == "__main__":
    try:
        # Argument parsing
        parser = argparse.ArgumentParser(description='Process GitHub issues.')
        parser.add_argument('repo_url', help='GitHub repository URL')
        parser.add_argument('max_issues', nargs='?', type=int, default=100, help='Maximum number of issues to fetch')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
        parser.add_argument('--disable-wordnet', action='store_true', help='Disable WordNet expansion of keywords')
        args = parser.parse_args()

        # Configure logging
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(message)s')

        repo_url = args.repo_url
        max_issues = args.max_issues

        # Get auth token from environment variable
        auth_token = os.getenv('GITHUB_TOKEN')
        if not auth_token:
            import getpass
            auth_token = getpass.getpass(
                "Enter GitHub Personal Access Token (https://github.com/settings/tokens):"
            )

        # Extract owner and repo name from the provided URL
        owner, repo_name = extract_owner_and_repo_from_url(repo_url)
        repo_owner = owner  # The owner of the repository

        # Get list of contributors/collaborators (excluding bots)
        logging.info("Fetching contributors...")
        contributors = get_contributors(auth_token, owner, repo_name)

        # Define cache file
        issues_cache_file = f"{repo_name}_issues_cache.json"

        # Load existing cached issues if available
        if os.path.exists(issues_cache_file):
            with open(issues_cache_file, 'r') as f:
                cached_issues = json.load(f)
            logging.info(f"Loaded {len(cached_issues)} issues from cache ({issues_cache_file}).")
        else:
            cached_issues = []
            logging.info("No cached issues found.")

        # Fetch issues
        logging.info("Fetching issues...")
        issues = get_issues(
            auth_token,
            owner,
            repo_name,
            states=['OPEN', 'CLOSED'],
            max_issues=max_issues
        )

        # Handle errors in get_issues
        if issues is None:
            sys.exit(1)

        logging.info(f"Fetched {len(issues)} issues.")

        # Update cached issues with new data
        all_issues = update_cached_issues(cached_issues, issues)

        # Save all issues back to cache
        with open(issues_cache_file, 'w') as f:
            json.dump(all_issues, f)
        logging.info(f"Updated cache with latest issues.")

        logging.info(f"Processing {len(all_issues)} issues for priority calculation...")

        # Define your initial keyword list
        initial_keywords = [
            "question", "how do i", "documentation", "help",
            "issue", "error", "problem", "support", "usage",
            "tutorial", "example", "explain", "guide", "instruction",
            "clarification", "how to", "usage example", "instructions",
            "best practice", "setup", "configuration", "installation",
            "integration", "faq", "troubleshoot", "explanation",
            "reference", "manual", "walkthrough", "tips", "tricks",
            "solution", "answer", "sample code", "guide me",
            "cannot", "can't", "doesn't work", "failing",
            "unexpected", "behavior", "question about", "newbie",
            "beginner", "help me", "problem with", "unsure",
            "confused", "need assistance"
        ]

        # Decide whether to use WordNet expansion
        if args.disable_wordnet:
            expanded_keywords = [lemmatizer.lemmatize(keyword.lower()) for keyword in initial_keywords]
            logging.info("WordNet expansion is disabled. Using initial keywords only.")
        else:
            # Ensure that the NLTK data is downloaded
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)  # For lemmatization
            nltk.download('punkt', quiet=True)    # For word_tokenize
            from nltk.corpus import wordnet as wn

            expanded_keywords = expand_keywords_with_synonyms(initial_keywords)
            logging.info("WordNet expansion is enabled. Keywords expanded.")

        # Process all issues (both cached and new) and calculate priority
        priority_issues = []

        for issue in all_issues:
            logging.debug(f"Processing issue: {issue['url']}")
            priority_score = calculate_priority(issue, contributors, repo_owner, expanded_keywords)
            if priority_score is not None:
                priority_issues.append(
                    {
                        "url": issue["url"],
                        "priority": priority_score,
                        "title": issue["title"],
                        "state": issue["state"],
                    }
                )
            else:
                logging.debug(f"Issue {issue['url']} was skipped due to exclusion criteria.")
                continue

        # Sort issues based on priority score
        priority_issues.sort(key=lambda x: x["priority"], reverse=True)

        if priority_issues:
            logging.info(f"Top priority issues for repository: {owner}/{repo_name}")
            for i in range(0, min(10, len(priority_issues))):
                logging.info(
                    "Issue: {url}, Priority Score: {priority:.2f}, Title: {title}, State: {state}".format(
                        **priority_issues[i]
                    )
                )
            # Save results to CSV file
            csv_file = f"{repo_name}_priority_issues.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['url', 'priority', 'title', 'state']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for issue in priority_issues:
                    writer.writerow(issue)
            logging.info(f"Results saved to {csv_file}")
        else:
            logging.info(f"No priority issues found in {owner}/{repo_name} after applying filters.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)