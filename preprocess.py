import re
import num2words
import textwrap

_MAXIMUM_ALLOWED_LENGTH = 140

def preprocess_text(text):
    '''
    Takes in string, replaces numbers with words, wraps the string into a list 
    of multiple lines.
    '''
    text = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), text)
    lines = textwrap.wrap(text, _MAXIMUM_ALLOWED_LENGTH, break_long_words=False)
    return lines