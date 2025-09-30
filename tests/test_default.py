from error_align import error_align, ErrorAlign
from error_align.utils import categorize_char, OpType


def test_error_align() -> None:
    """Test error alignment for an example including all substitution types."""

    ref = "This is a substitution test deleted."
    hyp = "Inserted this is a contribution test."

    alignments = error_align(ref, hyp, pbar=True)
    expected_ops = [
        OpType.INSERT,  # Inserted
        OpType.MATCH,  # This
        OpType.MATCH,  # is
        OpType.MATCH,  # a
        OpType.SUBSTITUTE,  # contribution -> substitution
        OpType.MATCH,  # test
        OpType.DELETE,  # deleted
    ]

    for op, alignment in zip(expected_ops, alignments):
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
    

def test_repr() -> None:
    """Test the string representation of Alignment objects."""

    # Test DELETE operation
    delete_alignment = error_align("deleted", "")[0]
    assert repr(delete_alignment) == 'Alignment(DELETE: "deleted")'

    # Test INSERT operation with compound markers
    insert_alignment = error_align("", "inserted")[0]
    assert repr(insert_alignment) == 'Alignment(INSERT: "inserted")'

    # Test SUBSTITUTE operation with compound markers
    substitute_alignment = error_align("substitution", "substitutiontesting")[0]
    assert substitute_alignment.left_compound is False
    assert substitute_alignment.right_compound is True
    assert repr(substitute_alignment) == 'Alignment(SUBSTITUTE: "substitution" -> "substitution"-)'

    # Test MATCH operation without compound markers
    match_alignment = error_align("test", "test")[0]
    assert repr(match_alignment) == 'Alignment(MATCH: "test" == "test")'
    
    # Test ErrorAlign class representation
    ea = ErrorAlign(ref="test", hyp="test")
    assert repr(ea) == 'ErrorAlign(ref="test", hyp="test")'
