from transformers import RobertaTokenizer, RobertaModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(True)
else:
    print(False)
    raise

batch_size = 64

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)


df = pd.read_csv('keras.csv')
code_snippets = df['function'].tolist()
code_tokens = tokenizer(code_snippets, padding=True, truncation=True, return_tensors="pt").to(device)
code_embeddings_list = []

for i in tqdm(range(0, len(code_snippets), batch_size)):
    batch_code_snippets = code_snippets[i:i+batch_size]
    code_tokens = tokenizer(batch_code_snippets, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        code_outputs = model(**code_tokens)
        code_embeddings = code_outputs.last_hidden_state[:, 0, :]
        code_embeddings_list.append(code_embeddings.cpu())

code_embeddings = torch.cat(code_embeddings_list, dim=0)
code_embeddings = code_embeddings.numpy()

with open('code_embeddings.pkl', 'rb') as f:
    code_embeddings = pickle.load(f)
np.save('code_embeddings.npy', code_embeddings)


generated_comments = ["""def get(identifier):
    if identifier is None:
        return linear
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    else:
        obj = identifier
    if callable(obj):
"""]

comment_embeddings_list = []

for i in range(0, len(generated_comments), batch_size):
    batch_comments = generated_comments[i:i+batch_size]
    comment_tokens = tokenizer(batch_comments, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        comment_outputs = model(**comment_tokens)
        comment_embeddings = comment_outputs.last_hidden_state[:, 0, :]
        comment_embeddings_list.append(comment_embeddings.cpu())


comment_embeddings = torch.cat(comment_embeddings_list, dim=0)
comment_embeddings = comment_embeddings.numpy()



similarity_scores = cosine_similarity(comment_embeddings, code_embeddings)
top_n = 5

for idx, comment in enumerate(generated_comments):
    sim_scores = similarity_scores[idx]
    top_indices = np.argsort(sim_scores)[::-1][:top_n]

    for rank, index in enumerate(top_indices):
        function_name = df.iloc[index]['identifier']
        score = sim_scores[index]
        print(f"{index}, {rank + 1}. function_name: {function_name}, similarity: {score:.4f}")
    print()