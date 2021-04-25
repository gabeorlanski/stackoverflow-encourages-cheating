from transformers import (
    PreTrainedTokenizer,
)
from typing import Optional, List, Dict, Any
import re
from .common import ObjectiveDataProcessor
from pathlib import Path
from ..asdl import asdl
from ..asdl.lang.py3 import py3_transition_system as py3_lang
import ast

__all__ = [
    "CodeGenerationProcessor"
]
double_newline = re.compile(r'\n\n+', re.MULTILINE)
double_space = re.compile(r'(\s)[^\S\n]+', re.MULTILINE)
markers = re.compile(r'(<\w+>)')


class CodeGenerationProcessor(ObjectiveDataProcessor):
    # TODO: Remove useless args eventually
    def __init__(self,
                 name: str,
                 model_name: str,
                 tokenizer: PreTrainedTokenizer,
                 html_tag_path: Path,
                 grammar_path: Path,
                 max_len: int = 512,
                 target_max_len: int = 128,
                 use_title_over_intent: bool = False,
                 use_body: bool = False,
                 remove_code_blocks: bool = False,
                 remove_inline_code: bool = False,
                 remove_all_code: bool = False,
                 use_only_code: bool = False,
                 disable_body_in_test: bool = False):
        columns_used = [
            'title' if use_title_over_intent else 'intent',
            'snippet',
            'body',
        ]

        #####################################################################
        # Parameters                                                        #
        #####################################################################
        self.use_title_over_intent = use_title_over_intent
        self.remove_code_blocks = remove_code_blocks or remove_all_code
        self.remove_inline_code = remove_inline_code or remove_all_code
        self.use_body = use_body
        self.use_only_code = use_only_code
        self.disable_body_in_test = disable_body_in_test
        if use_only_code:
            assert not remove_all_code

        # Helper dict for making naming and exporting parameters easier. Mainly
        # used for saving experiments. The first group above (starting from
        # `use_canonical_intent` to `use_title_over_intent`) are special and do
        # not go in this dict.
        self._parameters_to_shorthand = {
            'use_body'            : 'body',
            'use_only_code'       : 'onlyCode',
            'disable_body_in_test': 'noBodyTest',
        }

        # I should probably get around to adding the <> for these
        # automatically...
        special_tokens = [
            f"<{l.strip()}>" for l in html_tag_path.read_text('utf-8').splitlines(False)
            if l.strip()
        ]

        # Removed special tokens, but some legacy thing might have them, so add
        # those tokens to the special_tokens.
        special_tokens += [
            "<intent>",
            "<body>",
            "<snippet>"
        ]

        super(CodeGenerationProcessor, self).__init__(
            name,
            model_name,
            tokenizer,
            target_max_len,
            max_len,
            columns_used,
            special_tokens=special_tokens,
        )
        self._grammar = asdl.ASDLGrammar.from_text(open(grammar_path).read())
        self._transition_system = py3_lang.Python3TransitionSystem(self._grammar)

    def _parseExample(self,
                      intent: str,
                      snippet: str,
                      body: str,
                      in_test_data: bool,
                      **kwargs):

        tokenized = self.tokenizer.encode_plus(
            self.formatTargetString(
                intent,
                body if not in_test_data or (
                        in_test_data and not self.disable_body_in_test) else None,
            ),
            max_length=self.max_input_len,
            truncation=True
        )

        target = self.tokenizer.encode_plus(
            snippet,
            max_length=self.max_target_len,
            truncation=True
        )
        return {
            "input_ids"     : tokenized['input_ids'],
            "attention_mask": tokenized['attention_mask'],
            'labels'        : target['input_ids']
        }

    def formatTargetString(self, intent,
                           body=None) -> str:
        target_str = f"{intent} "

        if self.use_body:
            body_str = body
        else:
            body_str = None

        is_code_removed = self.remove_inline_code or self.remove_code_blocks
        if (
                body_str and (
                is_code_removed
                or self.use_only_code
        )):
            out = []
            prev_marker = None
            prev_start, prev_end = (0, 0)

            # Helper function to determine if we should keep the previous
            def shouldKeepSlice(end_span):

                is_marker_code = prev_marker in ['<code_block>', '<code>']
                if (
                        not is_marker_code
                ):
                    if not self.use_only_code:
                        # Only remove the token, but keep the contents
                        out.append(body_str[prev_end:end_span])
                elif (
                        (not is_marker_code and not self.use_only_code)
                        or
                        (prev_marker == '<code_block>' and not self.remove_code_blocks)
                        or
                        (prev_marker == '<code>' and not self.remove_inline_code)
                ):
                    # We can keep the entire slice of the string
                    out.append(body_str[prev_start:end_span])
                    if not is_marker_code:
                        out[-1] = double_space.sub(r'\1', out[-1])

            for match in filter(lambda m: m.group(0) in self.special_tokens,
                                markers.finditer(body_str)):
                current = match.group(0)
                current_start, current_end = match.span()

                # We do not consider 'console_in' and 'console_out' blocks their
                # own markers for this part.
                if current in ['<console_in>', '<console_out>']:
                    # We do set the previous marker to 'code_block' in case we
                    # are removing all code.
                    prev_marker = '<code_block>'
                    continue

                if prev_marker is not None:
                    shouldKeepSlice(end_span=current_start)
                elif current_start > 1:
                    # If Text before first marker, add it to out.
                    out.append(body_str[:current_start])

                prev_marker = current
                prev_start = current_start
                prev_end = current_end

            # prev_marker is None when no markers were found, which means there
            # was nothing to change, and therefore we can disregard everything
            # we did.
            if prev_marker is not None:
                # Have to handle final marker.
                shouldKeepSlice(len(body_str))
                target_str += double_newline.sub(r"\n", ''.join(out))
            else:
                target_str += body_str
        elif body_str:
            target_str += body_str
        for t in self.special_tokens:
            target_str = target_str.replace(t, '')
        return target_str.strip()

    @property
    def name(self):
        out = super(CodeGenerationProcessor, self).name

        if not self.use_title_over_intent:
            out += f"_Intent"
        else:
            out += '_title'

        out += f"_Snippet"

        if self.remove_code_blocks and self.remove_inline_code:
            out += f"_noCode"
        elif self.remove_code_blocks:
            out += f"_noBlocks"
        elif self.remove_inline_code:
            out += f"_noInline"
        for k, v in self._parameters_to_shorthand.items():
            if getattr(self, k):
                out += f"_{v}"

        return out

    @property
    def parameters(self):
        return {
            'name'                 : self._name,
            'model_name'           : self._model_name,
            'max_input_len'        : self.max_input_len,
            'max_target_len'       : self.max_target_len,
            'use_title_over_intent': self.use_title_over_intent,
            'remove_code_blocks'   : self.remove_code_blocks,
            'remove_inline_code'   : self.remove_inline_code,
            **{k: getattr(self, k) for k in self._parameters_to_shorthand.keys()}
        }
