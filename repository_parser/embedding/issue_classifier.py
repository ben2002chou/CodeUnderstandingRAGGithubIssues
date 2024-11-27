import os
from dotenv import load_dotenv
from openai import OpenAI
from openai_embedding import get_readme_description

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def classify_issue(local_dir, issue_title, issue_text):
    repository_description = get_readme_description(local_dir)
    prompt = f"""
You are a helpful ai that can classify a github issue post to the following categories:
1. The poster provided the code and they are using the repository as a tool like keras, pytorch. (example: Error when using model.fit() with custom data generator in TensorFlow/Keras)
2. The poster provided the code and they are implementing/editing the source code for their own use (example: I'm trying to modify the source code of Feature X in your project, but I've hit a roadblock.)
3. The code is not provided, and it's a general problem for the code repository. (example: environment setup and others)
4. This is something that can be related with a piece of specific code block in the repository, the poster just simply did not provide the code (example: Hi, I'm trying to use the processData function in one of the modules, but I'm running into an issue where it throws a ValueError when I pass certain inputs. )

Here is the repository description:
</description>
{repository_description}
</description>
Here is the issue post content:
</post> Title: {issue_title}
{issue_text}
</post>
You should only return the number of the category.
    """
    response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
            )
    return response.choices[0].message.content

