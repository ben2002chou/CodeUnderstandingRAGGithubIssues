import os
from dotenv import load_dotenv
from openai import OpenAI
from openai_embedding import get_readme_description

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def get_baseline_comment(function_body, repo_description):
    prompt = f"""
You are a helpful AI good at identifying underlying problem in a code. Identify possible issues user of this code may have. Here is the discussion of the repository where the code piece coming from:
</description>
{repo_description}
</description>
Here is the code:
</code>
{function_body}
</code>
"""
    response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
    return response.choices[0].message.content

if __name__ == "__main__":
    print(get_baseline_comment('''def stop():
    """Stop agent command"""
    import os
    import signal
    import subprocess

    try:
        pids = subprocess.check_output(["lsof", "-t", "-i", ":8000"]).split()
        if isinstance(pids, int):
            os.kill(int(pids), signal.SIGTERM)
        else:
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
    except subprocess.CalledProcessError:
        click.echo("No process is running on port 8000")

    try:
        pids = int(subprocess.check_output(["lsof", "-t", "-i", ":8080"]))
        if isinstance(pids, int):
            os.kill(int(pids), signal.SIGTERM)
        else:
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
    except subprocess.CalledProcessError:
        click.echo("No process is running on port 8080")''', "AutoGPT is a platform for creating, deploying, and managing AI agents that automate complex workflows. It offers a user-friendly interface for building custom agents, managing workflows, deploying agents, and monitoring performance. The repository includes components for the frontend, server, example agents, and classic version of AutoGPT. Users can self-host the platform or join a cloud-hosted beta program. The platform is under the MIT License with additional licensing information provided. "))