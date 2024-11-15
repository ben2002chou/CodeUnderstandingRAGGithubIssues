reference: https://github.com/github/CodeSearchNet


To parse a repository, first change the `\repository\parser.py` line 33 `folder_to_traverse = r'\path-to-repository'`, it will create a csv file.
put openai key in a .env file. Go to `\repository\embedding\openai_embedding.py`and change line 15 to the csv file, this will generate an embedding stored in csv. (provide a repository background information in line 18)
Then run `\repository\embedding\find_code_block.py`. The issue post content should be edited in line 16.