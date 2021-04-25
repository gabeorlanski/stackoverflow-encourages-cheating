from typing import Dict, List, Optional, Iterable
from pathlib import Path
import logging
from collections import Counter
from tqdm import tqdm
import sys
from src.common.debug_util import getBothLoggers
import json
from src.common.file_util import readJSONLFile
import re

import ast
from ...asdl.asdl import *
from ...asdl.lang.py.py_asdl_helper import *
from ...asdl.lang.py3.py3_transition_system import *
from ...asdl.hypothesis import *
import astor
from bs4 import BeautifulSoup
from unidecode import unidecode

__all__ = [
    "parseQuestionFiles",
    "StackOverflowQuestionParser"
]

# I am lazy man.

console_in = re.compile(r'(?:|^(?<=\n| ))(>>>|In ?\[[0-9]*\]:)', re.MULTILINE)

console_out = re.compile(r'(?<=\n| )(\.\.\.|Out ?\[[0-9]*\]:)', re.MULTILINE)
console_out_no_marker = re.compile(r'(<console_in>[^\n]*)\n ?(?!<)', re.MULTILINE)
add_space = re.compile(r"([^\s])<([^\s])", re.MULTILINE)


class StackOverflowQuestionParser:
    """
    Class for parsing StackOverflow questions
    """

    def __init__(self,
                 grammar_path: Path,
                 allow_python_tags: Optional[bool] = False,
                 logger: logging.Logger = None,
                 issue_logger: logging.Logger = None):

        if logger is None:
            self.logger, self.issue_logger = getBothLoggers()
        else:
            self.logger = logger
            self.issue_logger = issue_logger
        self.allow_python_tags = allow_python_tags
        # self.min_tag_count = min_tag_count
        self._tags = Counter()
        self.html_tags = Counter(['console_in', 'console_out'])
        self.grammar = ASDLGrammar.from_text(grammar_path.read_text('utf-8'))
        self.parser = Python3TransitionSystem(self.grammar)

    def __call__(self, questions: List[Dict]) -> List[Dict]:
        """

        Args:
            questions:

                Expected to follow the same format as the StackOverflow api
                because, you know, it is called StackOverflowQuestionParser.
                Yes, math, coffee, Русский язык, Mi Yodeya, Salesforce (pls hire
                me), Amateur Radio, or whatever *.stackexhange.com domain should
                work if the format is the same as stackoverflow.
        Returns:

        """
        self.logger.info(f"Parsing {len(questions)} questions")
        out = []

        for question in tqdm(questions, file=sys.stdout, desc="Parsing"):
            question_dict = {
                "question_id"       : str(question['question_id']),
                'tags'              : list(self._filterTags(question['tags'])),
                'title'             : question['title'].lower(),
                'accepted_answer_id': question[
                    'accepted_answer_id'] if 'accepted_answer_id' in question else None,
                'score'             : question['score']
            }

            body_parsed, code_slots = self._parseBody(question['body'])
            question_dict['body'] = body_parsed
            question_dict['code_slots'] = code_slots
            answers = {}
            try:
                for answer in question['answers']:
                    answer_parsed = {
                        'score': answer['score']
                    }
                    try:
                        answer_body, answer_code_slots = self._parseBody(answer['body'])
                    except:
                        continue
                    answer_parsed['body'] = answer_body
                    answer_parsed['code_slots'] = answer_code_slots
                    answers[str(answer['answer_id'])] = answer_parsed
            except KeyError:
                answers = {}
            question_dict['answers'] = answers
            out.append(question_dict)

        return out

    def _filterTags(self, tags: List[str]) -> Iterable[str]:
        for tag in tags:

            # TODO: This inadvertently removes the ability to have zero shot
            #  learning... might want to deal with this.
            # This handles both too uncommon tags and unknown tags. This also
            # automatically handles 'python' tags that are excluded during the
            # population stage because they will not be in the tag counter.
            if 'python' in tag.lower() and not self.allow_python_tags:
                continue

            yield tag.lower()

    def _parseBody(self,
                   html: str):

        # Get the body element from the HTML. This will be a bs4 Tag.
        soup = BeautifulSoup(html, 'lxml').find('body')

        out_strs = []
        # code_blocks = 0
        # code_spans = 0
        # paragraphs = 0
        code_slots = []
        for s in soup.strings:

            # We do not care if it is not a code element.
            if s.parent.name != 'code':
                parent = s.parent.name
            else:
                # We do not know if the parent of `s` has a parent or if it has
                # a name attr, therefore use a double getattr with defaults. Set
                # it to `'NOT_PRE'` because we are looking for `'pre'`.
                grand_parent = getattr(getattr(s.parent, 'parent', 'NOT_PRE'), 'name', 'NOT_PRE')

                # CodeBlocks on StackOverflow are grandchildren of a `<pre>`
                # tag. We want to mark code blocks, so we need to check for that
                # tag.
                if grand_parent == 'pre':
                    parent = 'code_block'
                else:
                    parent = s.parent.name

            # We need the list of unique tags, so I used a counter to keep track
            # of them.
            self.html_tags[parent] += 1

            # Sanitize the string.
            decoded = unidecode(s)

            # Empty string means new paragraph.
            if not decoded.strip():
                out_strs.append('\n')
                continue
            if 'code' in parent:
                # Replace the `'>>>'` with `'<console_in>'` and the output of the
                # console with `'<console_out>'`. Put in the extra quotes to
                # prevent the markers from being split in tokenization.
                code = console_in.sub(r"'<console_in>'", decoded.strip())
                code = console_out.sub(r"'<console_out>'", code)
                code = console_out_no_marker.sub(r"\1\n\'<console_out>'", code)

                # Try first to tokenize the entire block.
                try:
                    code_tokenized = self.parser.tokenize_code(code)

                except:
                    code_tokenized = None
                if code_tokenized:
                    # Remove the special markers on that we had to put in.
                    code_slots.append([
                        t.replace("'", '') if '<console_in>' in t or '<console_out>' in t else t
                        for t in code_tokenized])
                    # # Put a marker in to the output so we can replace later.
                    # out_strs.append(f"<{parent}> __CSLOT__")
                    # continue

            out_strs.append(f"<{parent}> {decoded}")

        # Repeat incase any tokenization of code ran into errors
        out = console_in.sub('<console_in>', ''.join(out_strs))
        out = console_out.sub('<console_out>', out)
        out = console_out_no_marker.sub(r'\1\n<console_out> ', out)
        out = add_space.sub(r'\1 <\2', out)
        return out, code_slots


def parseQuestionFiles(file_list: List[Path],
                       data_path: Path,
                       logger: logging.Logger,
                       issue_logger: logging.Logger):
    # Get the loggers
    logger.info(f"Parsing questions from {len(file_list)} files.")
    parser = StackOverflowQuestionParser(
        data_path.joinpath('py3_asdl.grammar'),
        logger=logger,
        issue_logger=issue_logger
    )

    questions = {}
    duplicates = Counter()
    tag_counts = Counter()

    # Go through every file in the list of files and read the questions.
    for file in file_list:

        file: Path

        # Only have support for json or jsonl
        if file.suffix != '.json' and file.suffix != '.jsonl':
            issue_logger.warning(f"'{file}' is not a supported file type")
            continue
        elif file.suffix == '.json':
            file_data = json.loads(file.read_text('utf-8'))

            # Sanity check really, just needs to be a list
            assert isinstance(file_data, list)
        else:
            file_data = readJSONLFile(file)

        logger.info(f"'{file}' has {len(file_data)} questions")

        # Parse the questions
        parsed = parser(file_data)

        # Go through and create the dictionary for the questions. We also check
        # for duplicates because that should not happen.
        for question in tqdm(parsed, file=sys.stdout, desc='Adding new Questions'):
            if question["question_id"] in questions:
                issue_logger.warning(f"{question['question_id']} is duplicated in '{file}'")
                duplicates[question['question_id']] += 1
                continue
            questions[str(question['question_id'])] = question
            tag_counts[len(question['tags'])] += 1

    logger.info(f"{sum(duplicates.values())} duplicates")
    logger.info(f"{len(questions)} questions parsed")
    logger.info(f"Tag Counts stats:")
    counts = list(tag_counts.items())
    for k, v in sorted(counts, key=lambda x: x[0]):
        pct_str = f"{v / len(questions) * 100:.2f}"
        logger.info(f"\t{k} = {v:>5}({pct_str:>6}%)")

    logger.info(
        f"{len(parser.html_tags)} html tags found, saving to "
        f"{data_path.joinpath('html_tags.txt')}")
    with data_path.joinpath('html_tags.txt').open('w', encoding='utf-8') as f:
        for t in parser.html_tags.keys():
            f.write(f"{t}\n")

    parsed_output = data_path.joinpath('parsed_so.json')
    logger.info(f"Writing {len(questions)} parsed questions to {parsed_output}")
    with parsed_output.open('w', encoding='utf-8') as fp:
        json.dump(questions, fp)


def run():
    test_questions = json.load(Path('example_question_format.json').open('r', encoding='utf-8'))
    # for q in test_questions:
    #     q:Dict


if __name__ == '__main__':
    run()
