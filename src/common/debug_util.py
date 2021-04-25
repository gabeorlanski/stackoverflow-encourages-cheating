import logging
import os


def logCLIArgs(name: str, logger: logging.Logger, **kwargs) -> None:
    """
    Helper function for logging the CLI args. Just pass in CLI args as keyword arguments,
    and it will log them.

    Examples:
        logCLIArgs('foo', logger, debug=debug, verbose=verbose, input_dir=input_dir)

    Args:
        name (str): the name of the program
        logger (logging.Logger): The logger to write to
        **kwargs:
    """
    logger.info(f"'{name}' called with command line arguments:")
    for var_name, value in kwargs.items():
        logger.info(f"{var_name:>32} = {value}")


def exceptionToReadableStr(e: Exception):
    return f"{type(e).__name__}:{str(e)}"


def validateHasAttr(obj: object, obj_name: str, attr_name: str, can_be_none: bool = False):
    # Check that the attribute exists. If it does not raise an error.
    if not hasattr(obj, attr_name):
        raise AttributeError(f"{obj_name} does not have attribute {attr_name}")

    # We know that the attribute exists, now we check if it is None. If the attribute is NOT
    # allowed to be None, raise an error.
    if getattr(obj, attr_name, None) is None and not can_be_none:
        raise ValueError(f"{obj_name}.{attr_name} is None")


def getBothLoggers():
    logger = logging.getLogger(os.environ.get('LOGGER_NAME', __name__))
    issue_logger = logging.getLogger(os.environ.get('ISSUE_LOGGER_NAME', __name__))
    return logger, issue_logger
