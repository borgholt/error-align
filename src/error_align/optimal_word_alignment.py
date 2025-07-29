from collections import defaultdict

import numpy as np
from tqdm import tqdm

from error_align.edit_distance import compute_optimal_word_alignment_distance_matrix
from error_align.optimal_alignment_graph import OptimalAlignmentGraph
from error_align.utils import (
    Alignment,
    basic_normalizer,
    basic_tokenizer,
    DELIMITERS,
    get_manhattan_distance,
    OpType,
    same_type_letter,
)


class OptimalWordAlign:
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

        # Initialize the optimal alignment graph from the backtrace matrix.
        _, B = compute_optimal_word_alignment_distance_matrix(self._ref, self._hyp, backtrace=True)
        self._optimal_alignment_graph = OptimalAlignmentGraph(B)

    def align(self) -> list[Alignment]:
        """
        Perform the optimal word alignment between the reference and hypothesis sequences.

        Returns:
            list[Alignment]: A list of Alignment objects representing the optimal word alignment.
        """
        path = self._optimal_alignment_graph.get_path()
        alignments = []
        for op_type, node in path:
            if op_type == OpType.MATCH or op_type == OpType.SUBSTITUTE:
                ref_match = self._ref_token_matches[node.ref_idx - 1]
                hyp_match = self._hyp_token_matches[node.hyp_idx - 1]
                alignment = Alignment(
                    op_type=op_type,
                    ref_slice=slice(*ref_match.span()),
                    hyp_slice= slice(*hyp_match.span()),
                    ref=ref_match.group(),
                    hyp=hyp_match.group(),
                    left_compound=False,
                    right_compound=False,
                )
            elif op_type == OpType.INSERT:
                ref_match = self._ref_token_matches[node.ref_idx - 1]
                alignment = Alignment(
                    op_type=op_type,
                    ref_slice=slice(*ref_match.span()),
                    ref=ref_match.group(),
                    left_compound=False,
                    right_compound=False,
                )
            elif op_type == OpType.DELETE:
                hyp_match = self._hyp_token_matches[node.hyp_idx - 1]
                alignment = Alignment(
                    op_type=op_type,
                    hyp_slice=slice(*hyp_match.span()),
                    hyp=hyp_match.group(),
                    left_compound=False,
                    right_compound=False,
                )
            else:
                raise ValueError(f"Unknown operation type: {op_type}")
            alignments.append(alignment)
        
        return alignments