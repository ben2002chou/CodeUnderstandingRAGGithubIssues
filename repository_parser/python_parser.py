from tree_sitter import Language, Parser
from typing import List, Dict, Any, Optional
import re
import logging
from colorama import Fore, Style, init
import tree_sitter_python

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
PYTHON_LANGUAGE = Language(tree_sitter_python.language())

class PythonParser:

    @staticmethod
    def get_definition(tree, blob: str) -> List[Dict[str, Any]]:
        definitions = PythonParser.traverse_tree(tree.root_node, blob)
        return definitions

    @staticmethod
    def traverse_tree(node, blob: str, class_name: Optional[str] = None, parent_function: Optional[str] = None) -> List[Dict[str, Any]]:
        definitions = []

        if node.type == 'class_definition':
            class_name_current = PythonParser.get_node_text(node.child_by_field_name('name'), blob)
            definitions.extend(PythonParser.process_class(node, blob))

            for child in node.children:
                definitions.extend(PythonParser.traverse_tree(child, blob, class_name=class_name_current, parent_function=parent_function))

        elif node.type == 'function_definition':
            func_name = PythonParser.get_node_text(node.child_by_field_name('name'), blob)
            identifier = func_name
            if class_name:
                identifier = f"{class_name}.{func_name}"
            if parent_function:
                identifier = f"{parent_function}.{func_name}"

            definitions.extend(PythonParser.process_function(node, blob, class_name, parent_function))

            for child in node.children:
                definitions.extend(PythonParser.traverse_tree(child, blob, class_name=class_name, parent_function=identifier))

        else:
            for child in node.children:
                definitions.extend(PythonParser.traverse_tree(child, blob, class_name=class_name, parent_function=parent_function))

        return definitions

    @staticmethod
    def process_class(node, blob: str) -> List[Dict[str, Any]]:
        class_name = PythonParser.get_node_text(node.child_by_field_name('name'), blob)
        class_docstring = PythonParser.get_docstring(node, blob)
        definition = {
            'identifier': class_name,
            'type': 'class',
            'docstring': class_docstring,
            'function': PythonParser.get_node_text(node, blob),
            'start_point': node.start_point,
            'end_point': node.end_point,
        }
        return [definition]


    @staticmethod
    def process_function(node, blob: str, class_name: Optional[str] = None, parent_function: Optional[str] = None) -> List[Dict[str, Any]]:
        func_name = PythonParser.get_node_text(node.child_by_field_name('name'), blob)
        docstring = PythonParser.get_docstring(node, blob)
        function_code = PythonParser.get_node_text(node, blob)
        identifier = func_name
        if class_name:
            identifier = f"{class_name}.{func_name}"
        if parent_function:
            identifier = f"{parent_function}.{func_name}"

        function_metadata = {
            'identifier': identifier,
            'type': 'function',
            'docstring': docstring,
            'function': function_code,
            'start_point': node.start_point,
            'end_point': node.end_point,
            'class_identifier': class_name,
        }

        return [function_metadata]

    @staticmethod
    def get_node_text(node, blob: str) -> str:
        if node is None:
            return ''
        # return blob[node.start_byte:node.end_byte]
        return blob[node.start_byte:node.end_byte].decode('utf-8')


    @staticmethod
    def get_docstring(node, blob: str) -> str:
        if node is None or node.child_by_field_name('body') is None:
            return ''
        body_node = node.child_by_field_name('body')
        if body_node.child_count == 0:
            return ''
        first_child = body_node.children[0]
        if first_child.type == 'expression_statement' and first_child.child_count > 0:
            expr = first_child.children[0]
            if expr.type == 'string':
                return PythonParser.get_node_text(expr, blob).strip('\"\' ')
        return ''


def log_definition(item):
    logger.info(Fore.BLUE + 'Identifier: ' + Style.RESET_ALL + (item['identifier'] or 'No Identifier'))
    if 'class_identifier' in item and item['class_identifier']:
        logger.info(Fore.MAGENTA + 'Class: ' + Style.RESET_ALL + item['class_identifier'])
    logger.info(Fore.YELLOW + 'Docstring: ' + Style.RESET_ALL + (item['docstring'] or 'No docstring available'))
    logger.info(Fore.GREEN + 'Code:\n' + Style.RESET_ALL + (item['function'] or 'None'))
    logger.info(Fore.BLUE + 'FilePath: ' + Style.RESET_ALL + item['f_path'])
    logger.info(Fore.RED + '-' * 60 + Style.RESET_ALL)
    





if __name__ == '__main__':
    code = '''

'''
    code_bytes = code.encode('utf-8')


    parser = Parser(PYTHON_LANGUAGE)
    # tree = parser.parse(bytes(code, 'utf8'))
    tree = parser.parse(code_bytes)
    definitions = PythonParser.get_definition(tree, code_bytes)

    for item in definitions:
        log_definition(item)
