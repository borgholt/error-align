import unicodedata
from dataclasses import dataclass
from enum import IntEnum
from itertools import chain
from itertools import combinations

import regex as re


class OpType(IntEnum):
    MATCH = 0
    INSERT = 1
    DELETE = 2
    SUBSTITUTE = 3
    
@dataclass
class Alignment:
    """Class representing an operation with its type and cost."""
    op_type: OpType
    ref_slice: slice | None = None
    hyp_slice: slice | None = None
    ref: str | None = None
    hyp: str | None = None
    left_compound: bool | None = None
    right_compound: bool | None = None
    
    @property
    def hyp_with_compound_markers(self) -> str:
        """Return the hypothesis with compound markers if applicable."""
        if self.hyp is None:
            return None
        return f"{'-' if self.left_compound else ''}{self.hyp}{'-' if self.right_compound else ''}"
        
    def __repr__(self):
        if self.op_type == OpType.INSERT:
            return f"Alignment({self.op_type.name}: {self.ref})"
        elif self.op_type == OpType.DELETE:
            return f"Alignment({self.op_type.name}: {self.hyp_with_compound_markers})"
        elif self.op_type == OpType.SUBSTITUTE:
            return f"Alignment({self.op_type.name}: {self.ref} -> {self.hyp_with_compound_markers})"
        else:
            return f"Alignment({self.op_type.name}: {self.ref} == {self.hyp_with_compound_markers})"


def op_type_powerset():
    """
    Generate all possible combinations of operation types, except the empty set.

    Returns:
        Generator: All possible combinations of operation types.
    """
    op_types = list(OpType)
    op_combinations = [combinations(op_types, r) for r in range(1, len(op_types) + 1)]
    return chain.from_iterable(op_combinations)


DELIMITERS = {"<", ">"}

OP_TYPE_MAP = {op_type.value: op_type for op_type in OpType}

OP_TYPE_COMBO_MAP = {i: op_types for i, op_types in enumerate(op_type_powerset())}

OP_TYPE_COMBO_MAP_INV = {v: k for k, v in OP_TYPE_COMBO_MAP.items()}


def normalize_char(c: str) -> str:
    """Normalize a character by removing accents.

    Args:
        c (str): The character to normalize.

    Returns:
        str: The normalized character in lowercase.
    """
    return unicodedata.normalize("NFD", c)[0].lower()


def is_vowel(c: str) -> bool:
    """Check if the normalized character is a vowel.

    Args:
        c (str): The character to check.

    Returns:
        bool: True if the character is a vowel, False otherwise.
    """
    return c in "aeiou"


def is_consonant(c: str) -> bool:
    """Check if the normalized character is a consonant.

    Args:
        c (str): The character to check.
    Returns:
        bool: True if the character is a consonant, False otherwise.
    """
    return c in "bcdfghjklmnpqrstvwxyz"


def same_type_letter(a: str, b: str) -> bool:
    """
    Returns True if both characters are either vowels or consonants,
    accounting for accented characters.

    Args:
        a (str): The first character.
        b (str): The second character.

    Returns:
        bool: True if both characters are of the same type (vowel or consonant),
              False otherwise.
    """
    if len(a) != 1 or len(b) != 1:
        raise ValueError("Both inputs must be single characters.")

    a = normalize_char(a)
    b = normalize_char(b)

    return (is_vowel(a) and is_vowel(b)) or (is_consonant(a) and is_consonant(b))


def get_manhattan_distance(a: tuple[int], b: tuple[int]) -> int:
    """
    Calculate the Manhattan distance between two points a and b.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def basic_tokenizer(text: str) -> list:
    """
    Default tokenizer that splits text into words based on whitespace.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list: A list of tokens (words).
    """
    return list(re.finditer(r"(\p{N}+([,.]\p{N}+)+)|([\p{L}\p{N}]+)", text, re.UNICODE | re.VERBOSE))


def basic_normalizer(text: str) -> str:
    """
    Default normalizer that converts text to lowercase and removes accents.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The normalized text.
    """
    return text.lower()
