from pathlib import Path
import json
import requests
from tqdm import tqdm
from ...common.debug_util import getBothLoggers
import logging
from typing import List, Dict, Any, Union, Optional
import sys
import time
from collections import defaultdict

__all__ = [
    "getDataFromAPI"
]


def getQuestionDetailURL(ids_get: List[str],
                         filter_use: str = "!3zl2.BHmiROa9.Phi") -> str:
    """
    Helper Function to create the StackOverflow questions api call with filter
    Args:
        ids_get: `List[str]`
            The list of question ids to get

        filter_use: `str`
            The SO api filter to use.

    Returns: `str`
        The URL to request.

    """
    return f'https://api.stackexchange.com/2.2/questions/' \
           f'{";".join(ids_get)}?order=desc&sort=activity&site=stackoverflow&filter={filter_use}'


def getQuestions(questions: List[Union[Dict, str, int]],
                 out_path: Optional[Path] = None,
                 out_name: Optional[str] = 'so_questions',
                 use_jsonl: Optional[bool] = False,
                 testing: Optional[bool] = False,
                 logger: logging.Logger = None,
                 issue_logger: logging.Logger = None, ) -> List[Dict]:
    """
    Function to request question data from the StackOverflow API. Stops when
    below a certain quota cutoff.

    Fair warning, they mention a backoff for throttling. They do NOT mention how
    it is returned to the user, so I kinda guessed and do not think my
    implementation works. Good luck.

    Args:
        questions: `List`
            List of questions. This can be either a list of question
            dictionaries or a list of question ids.
        out_path: `Optional[Path]`
            The path for where to save the resulting file that contains the
            responses from the API. A backup of an existing file will also be
            saved here.
        out_name: `Optional[str]`, default=`'so_questions'`
            The name of the file that will be saved. This is also used to check
            if there is an existing file with questions in order to reduce
            requests to the API.
        use_jsonl: `Optional[bool]`, default=False
            Save in .jsonl format vs .json
        testing: `Optional[bool]`, default=False
            Use testing mode. If so, no requests to the API will be made.
        logger: `logging.Logger`
            Logger who logs log worthy events and messages.
        issue_logger: `logging.Logger`
            Logger for logging log messages that the implementer deems too hot
            for the kettle.

    Returns: `List[Dict]`
        The list of dicts retrieved from the API.
    """
    if logger is None:
        logger, issue_logger = getBothLoggers()
    logger.info(
        f"Getting info for {len(questions)} question{'s' if len(questions) > 1 else ''} from "
        f"StackOverflow API")

    # Get the question ids from questions then remove duplicates
    all_questions = []
    for question in questions:
        if isinstance(question, dict):
            all_questions.append(str(question['question_id']))
        else:
            # Add the question id. Can be an int so convert to str.
            all_questions.append(str(question))
    all_questions = list(set(all_questions))
    logger.info(f"{len(all_questions)}/{len(questions)} are unique.")

    out_questions = []

    # StackOverflow limits API calls by using a quota for maximum number of
    # calls in a Day. For this program, since it is not authenticated, we are
    # limited to only 300 calls a day. To combat this, we warn the user when the
    # quota is low and try to avoid calling the same question more than once.
    quota = 300
    quota_warn_amount = 10
    max_ids_in_call = 30

    # Create the file name for outputting
    out_file = out_path.joinpath(f"{out_name}.{'jsonl' if use_jsonl else 'json'}")

    # Check if there already is a file with StackOverflow questions. First check
    # if there is a file with `out_file`.
    if out_file.exists():
        logger.debug(f"Found file at '{str(out_file)}'")

        # Set a variable to point to the existing file. Although we already have
        # `out_file` and we know it exists (only if we are here), we still need
        # to check if `out_file` does not exist.
        existing_file = out_file
    else:
        # Create the file for the other type whether that be `.json` or `.jsonl`
        possible_other_file_name = f"{out_name}.{'jsonl' if not use_jsonl else 'json'}"
        logger.debug(f"Did not find file at '{str(out_file)}' checking "
                     f"{possible_other_file_name}")

        # Create the path to check
        existing_file = out_path.joinpath(possible_other_file_name)

        # Check if it exists
        if existing_file.exists():
            logger.debug(f"Found file at '{existing_file}'")
        else:
            logger.info(f"Could not find any existing files to get already "
                        f"requested question data")
            # Set `existing_file=None` so that we know not to try and open it.
            existing_file = None

    # If we found an existing file.
    if existing_file is not None:
        logger.debug(f"Reading '{existing_file}'")

        # `existing_file` can point to either a `.json` or a `.jsonl`. Both
        # files have different processes for reading. Therefore, we check which
        # reading process to use.
        if existing_file.suffix == '.json':
            existing_questions: List = json.loads(existing_file.read_text('utf-8'))
        else:
            # We assume it is a `.jsonl` file. Therefore, we know that every
            # line is a `json` object and we can convert to a dict with json. We
            # also strip '\n' from each line and skip empty lines.
            existing_questions: List = [
                json.loads(line) for line in existing_file.read_text('utf-8').splitlines(False) if
                line.strip()]

        # Save the file to a backup
        with out_path.joinpath(
                f"backup_{out_name}{'.jsonl' if use_jsonl else '.json'}"
        ).open('w', encoding='utf-8') as f:
            if not use_jsonl:
                json.dump(existing_questions, f, indent=True)
            else:
                for question in existing_questions:
                    f.write(json.dumps(question) + '\n')
        # Iterate through each question and check that if it is in `questions`.
        # If it is, we remove it so as to reduce necessary calls. This also
        # serves as a VERY loose validation that the question dict has the
        # correct keys.
        failed = 0
        removed = 0
        new_questions = 0
        logger.info(f"Found {len(existing_questions)} already retrieved questions")
        for question in tqdm(existing_questions, file=sys.stdout, desc="Removing Parsed",
                             total=len(existing_questions)):
            question: Dict

            # Check if it has an id
            question_id = str(question.get('question_id', None))
            if question_id is None:
                failed += 1
                continue

            # Try to remove the question from all_questions
            try:
                all_questions.remove(question_id)
                removed += 1
            except ValueError:
                new_questions += 1
                logger.debug(f"{question_id} is not in all_questions")

            out_questions.append(question)

        logger.info(f"{len(out_questions)} questions had an ID.")
        logger.info(f"{failed}({failed / len(existing_questions) * 100:.2f}%)  failed")
        logger.info(
            f"{new_questions}({new_questions / len(existing_questions) * 100:.2f}%) were new "
            f"questions.")
        logger.info(
            f"{removed}({removed / len(existing_questions) * 100:.2f}%) were removed from "
            f"questions.")

    # Calculate the number of requests needed
    requests_count = len(all_questions) // max_ids_in_call
    if len(all_questions) % max_ids_in_call != 0:
        requests_count += 1

    logger.info(
        f"Getting question data for {len(all_questions)} questions will require {requests_count} "
        f"requests")

    if requests_count >= quota:
        issue_logger.warning(f"More requests than the StackOverflow API Daily Quota of {quota}")
        issue_logger.warning(f"Slicing down the number of questions to {quota - quota_warn_amount}")
        requests_count = quota - quota_warn_amount
        all_questions = all_questions[:max_ids_in_call * (quota - quota_warn_amount)]

    # I like to make the pbar outside of the
    pbar = tqdm(total=requests_count, file=sys.stdout, desc='Making API requests')

    # Iterate through `all_questions` and make the API calls to get question
    # data that will be saved.
    for i in range(0, len(all_questions), max_ids_in_call):
        ids_to_get = all_questions[i:i + max_ids_in_call]
        logger.debug(f"Getting questions {i} to {i + max_ids_in_call}")
        url = getQuestionDetailURL(ids_to_get)

        # Don't burn quota when testing.
        if testing:
            logger.info(f"Testing is enabled, so just skipping")
            pbar.update()
            continue

        try:
            request_data = json.loads(requests.get(url).text)
        except Exception as e:
            issue_logger.warning(f"Failed to get '{url[:100]}' with exception {e}")
            pbar.update()

            # Try to sleep to stop cooldown if there was a cooldown.
            time.sleep(15)

            continue

        # StackOverflow API does not like bombarding it. So if they tell us to
        # backoff, it best be that the program backs a few steps back in order
        # to follow their instructed backing off. God my hands hurt from typing
        # so many comments. I hope someone notices this garbage one haha. I got
        # a chuckle out of it at least.
        backoff = request_data.get('backoff', None)
        if backoff is not None:
            logger.info(f"Got backoff of {backoff}")
            time.sleep(int(backoff))
            break

        try:
            out_questions.extend(request_data['items'])
        except Exception as e:
            issue_logger.warning(f"Failed to add '{url[:100]}' with exception {e}")
            pbar.update()
            continue
        quota_remaining = request_data["quota_remaining"]
        if quota_remaining < quota_warn_amount:
            issue_logger.warning(f"Below quota warning amount of {quota_warn_amount}")
            break
        logger.info(f"{quota_remaining} quota remaining")
        pbar.update()

        # Try to not get in trouble
        time.sleep(.5)
    pbar.close()

    # Write the question dicts to the output file
    with out_file.open('w', encoding='utf-8') as fp:
        logger.info(f"Writing {len(out_questions)} questions to '{out_file}' (May take some time)")

        # If the file type is jsonl we write each line individually, else we can
        # just use the builtin json function to write.
        if not use_jsonl:
            json.dump(out_questions, fp, indent=True)
        else:
            for question in tqdm(out_questions, file=sys.stdout, desc=f"Writing data"):
                fp.write(json.dumps(question) + '\n')

    return out_questions


def getDataFromAPI(input_path: Path,
                   save_api_path: Path,
                   logger: logging.Logger,
                   issue_logger: logging.Logger,
                   in_testing: bool = False) -> None:
    """
    Function to get data from the api.

    Args:
        input_path: `Path`
            Path to where the json or jsonl files are stored. Every item in the
            files MUST have a question_id key.
        save_api_path: `Path`
            Path for saving the API results.
        logger: `logging.Logger`
            Logger who logs log worthy events and messages.
        issue_logger: `logging.Logger`
            Logger for logging log messages that the implementer deems too hot
            for the kettle.
        in_testing: `bool`
            We are in a test, so no cheating.

    Returns:
        Nothing.

    """
    logger.info(f"Getting question data from the StackOverflow API.")
    logger.debug(f"Using data from files at '{input_path}'")
    logger.debug(f"Saving api results to '{save_api_path}'")

    # Track the questions and question ids so that we can reealign the data
    # later. We use a defaultdict of defaultdict list so that we can also keep
    # track of what file the questions came from.
    question_ids_to_file = {}

    # Get all files in the input_path. Could not just check for .json or .jsonl.
    # So had to just get every file.
    for file in input_path.glob('*'):

        # Need to different methods for reading a json vs a jsonl file
        if file.suffix == '.json':
            file_data = json.loads(file.read_text('utf-8'))
        elif file.suffix == '.jsonl':
            file_data = [json.loads(line) for line in file.read_text('utf-8').splitlines(False)]
        else:
            logger.info(f"'{file}' has suffix I don't care for. Skipping.")
            continue
        logger.info(f"Processing '{file}'")

        # Get the question ids from the data in the files.
        for d in file_data:
            question_id = d['question_id']
            question_ids_to_file[str(question_id)] = str(file)

    assert len(question_ids_to_file) > 0

    # Get the question information such as body, answers, tags, etc from the
    # list of questions produced by the preprocessor. This question data will be
    # saved to the disk in order to reduce the need to re-request them from the
    # StackOverflow API.
    getQuestions(list(question_ids_to_file.keys()),
                 save_api_path,
                 use_jsonl=True,
                 testing=in_testing,
                 logger=logger,
                 issue_logger=issue_logger)
