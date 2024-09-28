from typing import List, Tuple, Callable
import re
from snorkel.labeling import labeling_function

def create_labeling_functions_from_regex(regex_patterns: List[Tuple[re.Pattern, str, int, int]]) -> List[Callable]:
    """
    Create labeling functions from a list of regex patterns.

    Args:
        regex_patterns (List[Tuple[re.Pattern, str, int, int]]): A list of tuples where each tuple contains:
            - pattern (re.Pattern): Compiled regex pattern to search for.
            - name (str): Name of the labeling function.
            - label (int): Label to return if the pattern matches.
            - label_else (int): Label to return if the pattern does not match.

    Returns:
        List[Callable]: A list of labeling functions.
    """
    labeling_functions = []
    for i, (pattern, name, match_label, else_label) in enumerate(regex_patterns):
        @labeling_function(name=f'lf_regex_{name}')
        def labeling_function_instance(x, pattern=pattern, match_label=match_label, else_label=else_label):
            # Return the match label if the pattern matches, otherwise return the else label
            return match_label if pattern.search(x.text) else else_label
        labeling_functions.append(labeling_function_instance)
    return labeling_functions

def int_to_alphabetic_string(n: int) -> str:
    """
    Convert an integer to a string using alphabetic characters (a, b, c, ..., z, aa, ab, ...).

    Args:
        n (int): The integer to convert.

    Returns:
        str: The corresponding alphabetic string.
    """
    result = []
    while n > 0:
        n -= 1
        # Calculate the current character and append it to the result list
        result.append(chr(n % 26 + ord('a')))
        # Move to the next character position
        n //= 26
    # Join the list into a string and return it
    return ''.join(result[::-1])