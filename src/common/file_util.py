from typing import List, Dict, Iterable, Union
from pathlib import Path
import json
import os
import logging
from .log_handlers import TQDMLoggingHandler, CompactFileHandler
from .debug_util import exceptionToReadableStr

__all__ = [
    "readJSONLFile",
    "setupLoggers",
    "strToPath"
]


def readJSONLFile(file_name: Union[str, Path]) -> List[Dict]:
    """
    Read a '.jsonl' file and create a list of dicts
    Args:
        file_name: `Union[str,Path]`
            The file to open
    Returns:
        The list of dictionaries read from the 'file_name'
    """
    lines = (
        open(file_name, 'r', encoding='utf-8').readlines() if isinstance(file_name, str) else
        file_name.read_text('utf-8').splitlines(False)
    )
    return [json.loads(line) for line in lines]


def setupLoggers(name: str, log_path: str = None, verbose: bool = False,
                 debug: bool = False) -> Iterable[logging.Logger]:
    """
    Setup the logger
    Args:
        name: Name of the logger
        log_path: Path to directory where the logs will be saved
        verbose: Enable Verbose
        debug: Enable Debug

    Returns:
        The loggers
    """
    # Load in the default paths for log_path
    log_path = os.path.join(log_path, 'logs') if log_path is not None else os.path.join(os.getcwd(),
                                                                                        'logs')

    # Validate the path and clear the existing log file
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    normal_file = os.path.join(log_path, f'{name}.log')
    error_file = os.path.join(log_path, f'{name}.issues.log')
    with open(normal_file, 'w', encoding='utf-8') as f:
        f.write('')

    # The different message formats to use
    msg_format = logging.Formatter(fmt='%(message)s')
    verbose_format = logging.Formatter(fmt='[%(asctime)s - %(levelname)8s] %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
    error_format = logging.Formatter(
        fmt='[%(asctime)s - %(levelname)8s - %(name)20s - %(funcName)20s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # Create the file handler
    normal_file_handler = CompactFileHandler(normal_file, logging.DEBUG, verbose_format)
    error_file_handler = CompactFileHandler(error_file, logging.WARNING, error_format)

    # Setup the console handlers for normal and errors
    console_handler = TQDMLoggingHandler(logging.INFO if not debug else logging.DEBUG,
                                         fmt=msg_format if not verbose else verbose_format)
    error_handler = TQDMLoggingHandler(logging.WARNING, fmt=error_format)

    # Set the environment variable to the names of the logger for use in other parts of the program
    os.environ['LOGGER_NAME'] = name
    os.environ['ISSUE_LOGGER_NAME'] = f'{name}.issue'

    # Create and register the two loggers
    logger = logging.getLogger(name)
    logger.addHandler(console_handler)
    logger.addHandler(normal_file_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    issue_logger = logging.getLogger(f'{name}.issue')
    issue_logger.addHandler(error_handler)
    issue_logger.addHandler(normal_file_handler)
    issue_logger.addHandler(error_file_handler)
    issue_logger.setLevel(logging.WARNING)
    issue_logger.propagate = False

    return logger, issue_logger


def strToPath(file_path: str) -> Path:
    path_split = file_path.split(os.path.sep)
    if len(path_split) == 1:
        path_split = path_split[0].split(os.path.altsep)
    return Path(*path_split)


def loadJSONTypeFile(file_path: Union[str, Path]) -> List[Dict]:
    """
    Load the data from either a `.jsonl` or `.json` file.
    Args:
        file_path (str): Path to the file that will be loaded. Only supports '.json' and
        '.jsonl' files

    Returns:
        The list of dictionaries
    """
    # Prefer to deal with `Path` Objects
    if isinstance(file_path, str):
        file_path = strToPath(file_path)

    # Only support 'json' and 'jsonl' for now
    if file_path.suffix == '.json':
        # Could not figure out how to store the other open arguments without using either a
        # lambda function or partial. Went with lambda to reduce the number of imports.
        reader = lambda x: json.loads(file_path.read_text('utf-8'))
    elif file_path.suffix == '.jsonl':
        reader = readJSONLFile
    else:
        raise ValueError(f'{file_path.suffix} is not a supported file type')

    # Do the reading here so that we can catch and report any errors encountered.
    try:
        return reader(file_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not open '{file_path}', got exception {exceptionToReadableStr(e)}")
