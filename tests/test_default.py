from error_align import error_align
from error_align.utils import categorize_char, OpType


def test_error_align() -> None:
    """Test error alignment for an example including all substitution types."""

    ref = "This is a substitution test deleted."
    hyp = "Inserted this is a inclusion test."

    alignments = error_align(ref, hyp)
    expected_ops = [
        OpType.INSERT,  # Inserted
        OpType.MATCH,  # This
        OpType.MATCH,  # is
        OpType.MATCH,  # a
        OpType.SUBSTITUTE,  # inclusion -> substitution
        OpType.MATCH,  # test
        OpType.DELETE,  # deleted
    ]

    for op, alignment in zip(expected_ops, alignments):
        alignment.__repr__()
        assert alignment.op_type == op


def test_error_align_full_match() -> None:
    """Test error alignment for full match."""

    ref = "This is a test."
    hyp = "This is a test."

    alignments = error_align(ref, hyp)

    for alignment in alignments:
        assert alignment.op_type == OpType.MATCH


def test_categorize_char() -> None:
    """Test character categorization."""

    assert categorize_char("<") == 0 # Delimiters
    assert categorize_char("b") == 1 # Consonants
    assert categorize_char("a") == 2 # Vowels
    assert categorize_char("'") == 3 # Unvoiced characters
