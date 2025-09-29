from error_align import error_align
from error_align.utils import OpType

def test_error_align() -> None:
    """Test error alignment for matching examples."""
    
    ref = "This is a test."
    hyp = "This is a pest."
    
    alignments = error_align(ref, hyp)
    
    for i, alignment in enumerate(alignments):
        if i < 3:
            assert alignment.op_type == OpType.MATCH
        elif i == 3:
            assert alignment.op_type == OpType.SUBSTITUTE

