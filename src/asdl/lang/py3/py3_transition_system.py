# coding=utf-8

import ast

import astor

from ....asdl.lang.py.py_asdl_helper import asdl_ast_to_python_ast, python_ast_to_asdl_ast
from ....asdl.lang.py.py_utils import tokenize_code
from ....asdl.transition_system import TransitionSystem, GenTokenAction

# Change Author: Gabe Orlanski
# Had to import allennlp Registrable because common was not included
try:
    from common.registerable import Registrable
except ImportError:
    from ....common.registrable import Registrable


@Registrable.register('python3')
class Python3TransitionSystem(TransitionSystem):
    def tokenize_code(self, code, mode=None):
        return tokenize_code(code, mode)

    def surface_code_to_ast(self, code):
        py_ast = ast.parse(code)
        return python_ast_to_asdl_ast(py_ast, self.grammar)

    def ast_to_surface_code(self, asdl_ast):
        py_ast = asdl_ast_to_python_ast(asdl_ast, self.grammar)
        code = astor.to_source(py_ast).strip()

        if code.endswith(':'):
            code += ' pass'

        # first make sure the hypothesis code is parsable by `ast`
        # sometimes, the parser generates syntactically invalid surface code:
        # e.g., slot_0.1()
        # ast.parse(code)

        return code

    def compare_ast(self, hyp_ast, ref_ast):
        hyp_code = self.ast_to_surface_code(hyp_ast)
        ref_reformatted_code = self.ast_to_surface_code(ref_ast)

        ref_code_tokens = tokenize_code(ref_reformatted_code)
        hyp_code_tokens = tokenize_code(hyp_code)

        return ref_code_tokens == hyp_code_tokens

    def get_primitive_field_actions(self, realized_field):
        actions = []
        if realized_field.value is not None:
            if realized_field.cardinality == 'multiple':  # expr -> Global(identifier* names)
                field_values = realized_field.value
            else:
                field_values = [realized_field.value]

            tokens = []
            if realized_field.type.name == 'string':
                for field_val in field_values:
                    tokens.extend(field_val.split(' ') + ['</primitive>'])
            else:
                for field_val in field_values:
                    tokens.append(field_val)

            for tok in tokens:
                actions.append(GenTokenAction(tok))
        elif realized_field.type.name == 'singleton' and realized_field.value is None:
            # singleton can be None
            actions.append(GenTokenAction('None'))

        return actions

    def is_valid_hypothesis(self, hyp, **kwargs):
        try:
            hyp_code = self.ast_to_surface_code(hyp.tree)
            ast.parse(hyp_code)
            self.tokenize_code(hyp_code)
        except:
            return False
        return True
