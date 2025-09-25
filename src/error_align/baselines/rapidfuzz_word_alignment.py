from rapidfuzz.distance import Levenshtein

from error_align.edit_distance import compute_optimal_word_alignment_distance_matrix
from error_align.optimal_alignment_graph import OptimalAlignmentGraph
from error_align.utils import (
    Alignment,
    basic_normalizer,
    basic_tokenizer,
    OpType,
)

OPS_MAP = {
    "match": OpType.MATCH,
    "replace": OpType.SUBSTITUTE,
    "insert": OpType.INSERT,
    "delete": OpType.DELETE,
}


class RapidFuzzWordAlign:
    """
    Class to handle optimal word alignment in a sequence.
    """

    def __init__(
        self,
        ref: str,
        hyp: str,
        tokenizer: callable = basic_tokenizer,
        normalizer: callable = basic_normalizer,
    ):
        """
        Initialize the optimal word alignment with reference and hypothesis sequences.

        Args:
            ref (str): The reference sequence/transcript.
            hyp (str): The hypothesis sequence/transcript.
            tokenizer (callable): A function to tokenize the sequences. Must be regex-based and return Match objects.
            normalizer (callable): A function to normalize the tokens. Defaults to basic_normalizer.
        """
        if not isinstance(ref, str):
            raise TypeError("Reference sequence must be a string.")
        if not isinstance(hyp, str):
            raise TypeError("Hypothesis sequence must be a string.")

        self.ref = ref
        self.hyp = hyp
        self._ref_token_matches = tokenizer(ref)
        self._hyp_token_matches = tokenizer(hyp)
        self._ref = [normalizer(r.group()) for r in self._ref_token_matches]
        self._hyp = [normalizer(h.group()) for h in self._hyp_token_matches]
        self._ref_max_idx = len(self._ref) - 1
        self._hyp_max_idx = len(self._hyp) - 1
        self.end_index = (self._hyp_max_idx, self._ref_max_idx)

    def align(self) -> list[Alignment]:
        """
        Perform the optimal word alignment between the reference and hypothesis sequences.

        Returns:
            list[Alignment]: A list of Alignment objects representing the optimal word alignment.
        """
        edit_ops = Levenshtein.editops(self._hyp, self._ref).as_list()

        # Add match segments to editops
        ref_edit_idxs = set([op[2] for op in edit_ops if op[0] != "delete"])
        hyp_edit_idxs = set([op[1] for op in edit_ops if op[0] != "insert"])
        ref_match_idxs = [i for i in range(len(self._ref)) if i not in ref_edit_idxs]
        hyp_match_idxs = [i for i in range(len(self._hyp)) if i not in hyp_edit_idxs]
        assert len(ref_match_idxs) == len(hyp_match_idxs)
        for hyp_idx, ref_idx in zip(hyp_match_idxs, ref_match_idxs):
            edit_ops.append(("match", hyp_idx, ref_idx))
        edit_ops = sorted(edit_ops, key=lambda x: (x[2], x[1]))

        # Convert to Alignment objects
        alignments = []
        for op_type, hyp_idx, ref_idx in edit_ops:
            if op_type == "match" or op_type == "replace":
                ref_match = self._ref_token_matches[ref_idx]
                hyp_match = self._hyp_token_matches[hyp_idx]
                alignment = Alignment(
                    op_type=OPS_MAP[op_type],
                    ref_slice=slice(*ref_match.span()),
                    hyp_slice=slice(*hyp_match.span()),
                    ref=ref_match.group(),
                    hyp=hyp_match.group(),
                    left_compound=False,
                    right_compound=False,
                )
            elif op_type == "insert":
                ref_match = self._ref_token_matches[ref_idx]
                alignment = Alignment(
                    op_type=OPS_MAP[op_type],
                    ref_slice=slice(*ref_match.span()),
                    ref=ref_match.group(),
                    left_compound=False,
                    right_compound=False,
                )
            elif op_type == "delete":
                hyp_match = self._hyp_token_matches[hyp_idx]
                alignment = Alignment(
                    op_type=OPS_MAP[op_type],
                    hyp_slice=slice(*hyp_match.span()),
                    hyp=hyp_match.group(),
                    left_compound=False,
                    right_compound=False,
                )
            else:
                raise ValueError(f"Unknown operation type: {op_type}")
            alignments.append(alignment)

        return alignments
