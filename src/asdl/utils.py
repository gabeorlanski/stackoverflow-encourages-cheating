# coding=utf-8
import re


def remove_comment(text):
    # Change Author: Gabe Orlanski
    # Python 3.8 Abstract grammar uses -- as comments instead of #
    text = re.sub(re.compile("[#-].*"), "", text)
    text = '\n'.join(filter(lambda x: x, text.split('\n')))

    return text
