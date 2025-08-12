from collections import defaultdict

import numpy as np
from tqdm import tqdm

from error_align.edit_distance import compute_error_align_distance_matrix
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


class ErrorAlign:
    """
    Class to handle error alignment in a sequence.
    """

    def __init__(
        self,
        ref: str,
        hyp: str,
        tokenizer: callable = basic_tokenizer,
        normalizer: callable = basic_normalizer,
    ):
        """
        Initialize the error alignment with reference and hypothesis sequences.

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
        self._ref = "".join([f"<{normalizer(r.group())}>" for r in self._ref_token_matches])
        self._hyp = "".join([f"<{normalizer(h.group())}>" for h in self._hyp_token_matches])
        self._ref_max_idx = len(self._ref) - 1
        self._hyp_max_idx = len(self._hyp) - 1
        self.end_index = (self._hyp_max_idx, self._ref_max_idx)

        # Create index maps for reference and hypothesis sequences.
        # NOTE: -1 is used for delimiter (<>) and indicates no match in the source sequence.
        self._ref_index_map = np.full((len(self._ref),), -1, dtype=int)
        self._hyp_index_map = np.full((len(self._hyp),), -1, dtype=int)
        start = 1
        for match in self._ref_token_matches:
            end = start + len(match.group())
            self._ref_index_map[start:end] = np.arange(*match.span(), dtype=int)
            start = end + 2
        start = 1
        for match in self._hyp_token_matches:
            end = start + len(match.group())
            self._hyp_index_map[start:end] = np.arange(*match.span(), dtype=int)
            start = end + 2

        # Initialize the optimal alignment graph from the backtrace matrix.
        _, B = compute_error_align_distance_matrix(self._ref, self._hyp, backtrace=True)
        self._optimal_alignment_graph = OptimalAlignmentGraph(B)
        self._oga_node_indices = self._optimal_alignment_graph.get_node_indices()

    def align(self, beam_size=100, pbar=True) -> list[Alignment]:
        """
        Perform the alignment process.
        """
        # Initialize the beam with the starting path.
        start_path = Path(self)
        beam = {}
        beam[start_path.diversity_id] = start_path
        ended = []
        diversity_map = defaultdict(lambda: float("inf"))
        paths_pruned = 0

        # Iterate through the paths until we reach the end of both sequences.
        total_mdist = self._ref_max_idx + self._hyp_max_idx + 2
        if pbar:
            progress_bar = tqdm(total=total_mdist, desc="Aligning transcripts")
        while len(beam) > 0:
            for i, path in enumerate(beam.values()):  # DELETE
                path._ranks.append(i)  # DELETE: Store ranks for debugging
            new_beam = {}

            # Expand each path in the current beam.
            for path in beam.values():
                if path.at_end:
                    ended.append(path)
                    continue

                expanded_paths = path.expand()
                for new_path in expanded_paths:

                    if new_path.diversity_id in diversity_map:
                        if new_path.score > diversity_map[new_path.diversity_id]:
                            paths_pruned += 1
                            continue
                    diversity_map[new_path.diversity_id] = new_path.score

                    if new_path.diversity_id not in new_beam:
                        new_beam[new_path.diversity_id] = new_path
                    elif new_path.score < new_beam[new_path.diversity_id].score:
                        new_beam[new_path.diversity_id] = new_path
                    else:
                        paths_pruned += 1  # Diversity pruning

            # Update the beam with the newly expanded paths.
            new_beam = list(new_beam.values())
            new_beam.sort(key=lambda p: p.norm_score)
            beam = new_beam[:beam_size]

            # Keep only the best path if it matches the segment
            if len(beam) > 0 and beam[0]._match_segment:
                beam = beam[:1]
                diversity_map = defaultdict(lambda: float("inf"))

            beam = {p.diversity_id: p for p in beam}  # Convert to dict for diversity check

            try:
                worst_path = next(reversed(beam.values()))
                mdist = get_manhattan_distance(worst_path.index, self.end_index)
                if pbar:
                    progress_bar.n = total_mdist - mdist
                    progress_bar.refresh()
            except StopIteration:
                if pbar:
                    progress_bar.n = total_mdist
                    progress_bar.refresh()

        # print(f"### END: {len(ended)} paths ended, {paths_pruned} diversity pruned")
        ended.sort(key=lambda p: p.score)
        return ended[0].alignments if len(ended) > 0 else []


class Path:
    """
    Class to represent a file path.
    """

    def __init__(self, src: ErrorAlign):
        """
        Initialize the Path class with a given path.
        """
        self.ref_idx = -1
        self.hyp_idx = -1
        self.src = src
        self._path = []
        self._path_score = 0
        self._segment_score = 0
        self._hyp_segment_len = 0
        self._ref_segment_len = 0
        self._hyp_segment_start = 0
        self._ref_segment_start = 0
        self._alignments = []
        self._token_ops = []
        self._match_segment = False

        self._ranks = []  # DELETE

    def expand(self):
        """
        Expand the path by adding possible operations.
        """
        if self.at_end:
            return [self]

        new_paths = []

        # Add delete operation
        delete_path = self._add_delete()
        if delete_path is not None:
            new_paths.append(delete_path)

        # Add insert operation
        insert_path = self._add_insert()
        if insert_path is not None:
            new_paths.append(insert_path)

        # Add substitution or match operation
        sub_or_match_path = self._add_substitution_or_match()
        if sub_or_match_path is not None:
            new_paths.append(sub_or_match_path)

        return new_paths

    def _shallow_copy(self):
        """
        Create a shallow copy of the path.
        """
        new_path = Path(self.src)
        new_path.ref_idx = self.ref_idx
        new_path.hyp_idx = self.hyp_idx

        new_path._path = self._path.copy()
        new_path._path_score = self._path_score
        new_path._segment_score = self._segment_score
        new_path._hyp_segment_len = self._hyp_segment_len
        new_path._ref_segment_len = self._ref_segment_len
        new_path._hyp_segment_start = self._hyp_segment_start
        new_path._ref_segment_start = self._ref_segment_start
        new_path._alignments = self._alignments.copy()
        new_path._token_ops = self._token_ops.copy()
        new_path._match_segment = False

        new_path._ranks = self._ranks.copy()  # DELETE

        return new_path

    def _end_segment(self) -> None:
        """
        End the current segment of the path.
        """

        # Handle insertions
        if self._hyp_segment_len == 0:
            ref_slice = self.ref_segment_slice_original
            if ref_slice is None:
                return False
            alignment = Alignment(
                op_type=OpType.INSERT,
                ref_slice=ref_slice,
                ref=self.src.ref[ref_slice],
            )
            self._alignments.append(alignment)
            self._token_ops.append(OpType.INSERT)
            
        # Handle deletions
        elif self._ref_segment_len == 0:
            hyp_slice = self.hyp_segment_slice_original
            if hyp_slice is None:
                return False
            alignment = Alignment(
                op_type=OpType.DELETE,
                hyp_slice=hyp_slice,
                hyp=self.src.hyp[hyp_slice],
                left_compound=self.left_compound,
                right_compound=self.right_compound,
            )
            self._alignments.append(alignment)
            self._token_ops.append(OpType.DELETE)
        
        # Handle substitutions or matches
        else:
            ref_slice = self.ref_segment_slice_original
            hyp_slice = self.hyp_segment_slice_original
            if ref_slice is None or hyp_slice is None:
                return False
            self._match_segment = self.ref_segment == self.hyp_segment
            op_type = OpType.MATCH if self._match_segment else OpType.SUBSTITUTE
            alignment = Alignment(
                op_type=op_type,
                ref_slice=ref_slice,
                hyp_slice=hyp_slice,
                ref=self.src.ref[ref_slice],
                hyp=self.src.hyp[hyp_slice],
                left_compound=self.left_compound,
                right_compound=self.right_compound,
            )
            self._alignments.append(alignment)
            self._token_ops.append(op_type)

        # Update the path score and reset segments attributes.
        self._path_score += self.segment_score
        self._segment_score = 0
        self._ref_segment_start += self._ref_segment_len
        self._hyp_segment_start += self._hyp_segment_len
        self._hyp_segment_len = 0
        self._ref_segment_len = 0

        return True

    def _is_optimal_node(self, index):
        """
        Check if the given operation is an optimal transition at the current index.
        """
        return index in self.src._oga_node_indices

    def _add_delete(self):
        """
        Expand the given path by adding a delete operation.
        """
        if self.hyp_idx >= self.src._hyp_max_idx:
            return None
        new_path = self._shallow_copy()

        is_optimal = self._is_optimal_node(new_path.index)
        new_path._path.append(OpType.DELETE)
        new_path.hyp_idx += 1
        new_path._hyp_segment_len += 1

        is_delimiter = self.src._hyp[new_path.hyp_idx] in DELIMITERS
        new_path._segment_score += 1 if is_delimiter else 2
        new_path._segment_score += 0 if is_optimal and not is_delimiter else 1

        if new_path._ref_segment_len == 0 and self.src._hyp[new_path.hyp_idx] == ">":
            valid_segment = new_path._end_segment()
            if not valid_segment:
                return None

        return new_path

    def _add_insert(self):
        """
        Expand the given path by adding an insert operation.
        """
        if self.ref_idx >= self.src._ref_max_idx:
            return None
        new_path = self._shallow_copy()

        is_optimal = self._is_optimal_node(new_path.index)
        new_path._path.append(OpType.INSERT)
        new_path.ref_idx += 1
        new_path._ref_segment_len += 1

        is_delimiter = self.src._ref[new_path.ref_idx] in DELIMITERS
        new_path._segment_score += 1 if is_delimiter else 2
        new_path._segment_score += 0 if is_optimal and not is_delimiter else 1

        if self.src._ref[new_path.ref_idx] == ">":
            valid_segment = new_path._end_segment()
            if not valid_segment:
                return None

        return new_path

    def _add_substitution_or_match(self):
        """
        Expand the given path by adding a substitution or match operation.
        """
        if self.ref_idx >= self.src._ref_max_idx or self.hyp_idx >= self.src._hyp_max_idx:
            return None
        new_path = self._shallow_copy()

        is_match = self.src._ref[new_path.ref_idx + 1] == self.src._hyp[new_path.hyp_idx + 1]
        ref_is_delimiter = self.src._ref[new_path.ref_idx + 1] in DELIMITERS
        hyp_is_delimiter = self.src._hyp[new_path.hyp_idx + 1] in DELIMITERS
        if not is_match and (ref_is_delimiter or hyp_is_delimiter):
            return None

        op_type = OpType.MATCH if is_match else OpType.SUBSTITUTE
        is_optimal = self._is_optimal_node(new_path.index)
        new_path._path.append(op_type)
        new_path.ref_idx += 1
        new_path.hyp_idx += 1
        new_path._hyp_segment_len += 1
        new_path._ref_segment_len += 1

        if not is_match:
            is_letter_type_match = same_type_letter(self.src._ref[new_path.ref_idx], self.src._hyp[new_path.hyp_idx])
            new_path._segment_score += 2 if is_letter_type_match else 3
            new_path._segment_score += 0 if is_optimal else 1

        if self.src._ref[new_path.ref_idx] == ">":
            valid_segment = new_path._end_segment()
            if not valid_segment:
                return None

        return new_path

    def _get_original_slice(self, segment_slice: slice, index_map: np.ndarray):
        """
        Get the original slice from the segment slice using the index map.
        """
        slice_indices = index_map[segment_slice]
        slice_indices = slice_indices[slice_indices >= 0]
        if len(slice_indices) == 0:
            return None
        start, end = int(slice_indices[0]), int(slice_indices[-1] + 1)
        return slice(start, end)

    @property
    def alignments(self):
        """
        Get the alignments of the path.
        """
        return self._alignments

    @property
    def left_compound(self):
        """
        Check if the left side of the path is compound.
        """
        return self.src._hyp_index_map[self._hyp_segment_start] >= 0

    @property
    def right_compound(self):
        """
        Check if the right side of the path is compound.
        """
        return self.src._hyp_index_map[self._hyp_segment_start + self._hyp_segment_len - 1] >= 0

    @property
    def hyp_segment_slice(self):
        """
        Get the slice of the hypothesis segment.
        """
        return slice(self._hyp_segment_start, self._hyp_segment_start + self._hyp_segment_len)

    @property
    def ref_segment_slice(self):
        """
        Get the slice of the reference segment.
        """
        return slice(self._ref_segment_start, self._ref_segment_start + self._ref_segment_len)

    @property
    def ref_segment_slice_original(self):
        """
        Get the original slice of the reference segment.
        """
        return self._get_original_slice(self.ref_segment_slice, self.src._ref_index_map)

    @property
    def hyp_segment_slice_original(self):
        """
        Get the original slice of the hypothesis segment.
        """
        return self._get_original_slice(self.hyp_segment_slice, self.src._hyp_index_map)

    @property
    def hyp_segment(self):
        """
        Get the hypothesis segment.
        """
        return self.src._hyp[self.hyp_segment_slice]

    @property
    def ref_segment(self):
        """
        Get the reference segment.
        """
        return self.src._ref[self.ref_segment_slice]

    @property
    def segment_len_penalty(self):
        """
        Get the segment length penalty.
        """
        if min(self._hyp_segment_len, self._ref_segment_len) > 0:
            return self._segment_score
        return 0

    @property
    def diversity_id(self):
        """
        Get the diversity ID of the path.
        This is a unique identifier based on the path's index and the last 20 operations.
        """
        _is_substitution = min(self._ref_segment_len, self._hyp_segment_len) > 0
        return hash((self.index + tuple(self._token_ops) + (1 if _is_substitution else 0,)))

    @property
    def segment_score(self):
        """
        Get the segment score of the path.
        """
        return self._segment_score + self.segment_len_penalty

    @property
    def score(self):
        """
        Get the score of the path.
        """
        return self._path_score + self.segment_score

    @property
    def norm_score(self):
        """
        Get the normalized score of the path.
        """
        if self.score == 0:
            return 0
        return self.score / (self.ref_idx + self.hyp_idx + 3)  # +3 to avoid division by zero

    @property
    def index(self):
        """
        Get the index of the path.
        """
        return (self.hyp_idx, self.ref_idx)

    @property
    def at_end(self):
        """
        Check if the path has reached the end of both sequences.
        """
        return self.index == self.src.end_index

    def __repr__(self):
        """
        String representation of the Path object.
        """
        return f"Path(({self.ref_idx}, {self.hyp_idx}), score={self.score})"
