import os
import pandas as pd
from tree_sitter import Language, Parser
import tree_sitter_python
from python_parser import *

PYTHON_LANGUAGE = Language(tree_sitter_python.language())
def traverse_and_parse(folder_path: str):
    parser = Parser(PYTHON_LANGUAGE)
    all_definitions = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    code_bytes = code.encode('utf-8')
                    tree = parser.parse(code_bytes)
                    definitions = PythonParser.get_definition(tree, code_bytes)
                    for item in definitions:
                        # 添加文件路径信息
                        relative_path = os.path.relpath(file_path, folder_path)
                        item['f_path'] = relative_path.replace('\\', '/')
                        all_definitions.append(item)
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")

    return all_definitions

if __name__ == '__main__':
    folder_to_traverse = r'D:\academic\keras'
    prj_name = 'keras'
    definitions = traverse_and_parse(folder_to_traverse)

    df = pd.DataFrame(definitions)

    output_csv_path = '{}.csv'.format(prj_name)
    df.to_csv(output_csv_path, index=False)
    print(df.head())
