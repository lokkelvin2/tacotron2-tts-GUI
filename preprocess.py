import re
import num2words
import textwrap

_MAXIMUM_ALLOWED_LENGTH = 140


def add_fullstop(text):
    out = text+'.' if text[-1]!='.' else text
    return out

def break_long_sentences(text):
    return textwrap.wrap(text, _MAXIMUM_ALLOWED_LENGTH, break_long_words=False)

def preprocess_text(text):
    '''
    Takes in string, replaces numbers with words, wraps the string into a list 
    of multiple lines.
    '''
    text = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), text)
    lines = break_long_sentences(text)
    if lines:
        lines[-1] = add_fullstop(lines[-1])
    return lines