import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


df = pd.read_csv("./A-LLMRec.csv")
df['start_row'] = df['start_point'].str.extract(r'row=(\d+)')
df['end_row'] = df['end_point'].str.extract(r'row=(\d+)')
df['function_for_LLM'] = "This is a function from A-LLMRec Repository, which is a LLM for recommendation system program. The file path is ./" + df['f_path'] + "." + "The function starts at line " + df['start_row'] + "and ends at line " + df['end_point'] + "<function_body>" + df['function'] + "<function_body>"
df['embedding'] = df['function_for_LLM'].apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
df.to_csv('output/embedded.csv', index=False)
