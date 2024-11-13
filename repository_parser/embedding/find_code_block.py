from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
load_dotenv()
from scipy import spatial  # for calculating vector similarities for search
import ast

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful summarizing AI."},
        {
            "role": "user",
            "content": """This is a github issue post content in A-LLMRec repository, which is a LLM for recommendation system. Please try to understand the content, extract important information, generate an appropriate, helpful, brief comment that will be placed directly in the code file. Here is the content:
            \<content>
            [title] RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::Half != float 
            [person 1] When I run stage 2. The error happened. Has anyone meet the same error?
            [person 2] I often faced that issue due to the quantizing of the LLM model. From below code in the llm4rec.py, try removing the load_in_8bit=True option.
            self.llm_model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16, load_in_8bit=True, device_map=self.device)
            [person 1] Thank you! I have done
            """
        }
    ]
)

print(completion.choices[0].message.content)
df = pd.read_csv("./output/embedded.csv")
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
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

def generate_answer(question, related_function):
    prompt = f"Question: {question}\n\nContext:\n{related_function}\n\nAnswer:"
    response = OpenAI.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

question = completion.choices[0].message.content
strings, relatednesses = strings_ranked_by_relatedness(question, df, top_n=5)
for string, relatedness in zip(strings, relatednesses):
    print(f"{relatedness=:.3f}")
    print(string)