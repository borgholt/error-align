from error_align.baselines.power.power.aligner import PowerAligner as _PowerAligner
from error_align.utils import Alignment, OpType


class PowerAlign:
    """
    Class to handle optimal word alignment in a sequence.
    """

    def __init__(
        self,
        ref: str,
        hyp: str,
    ):
        """
        Initialize the optimal word alignment with reference and hypothesis sequences.

        Args:
            ref (str): The reference sequence/transcript.
            hyp (str): The hypothesis sequence/transcript.
            tokenizer (callable): A function to tokenize the sequences. Must be regex-based and return Match objects.
            normalizer (callable): A function to normalize the tokens. Defaults to basic_normalizer.
        """
        self.aligner = _PowerAligner(
            ref=ref,
            hyp=hyp,
            lowercase=True,
            verbose=True,
            lexicon="/home/lb/repos/power-asr/lex/cmudict.rep.json",
        )

    def align(self):
        """
        Extract alignments from the PowerAligner instance.
        """
        self.aligner.align()
        widths = [
            max(len(self.aligner.power_alignment.s1[i]), len(self.aligner.power_alignment.s2[i]))
            for i in range(len(self.aligner.power_alignment.s1))
        ]
        s1_args = list(zip(widths, self.aligner.power_alignment.s1))
        s2_args = list(zip(widths, self.aligner.power_alignment.s2))
        align_args = list(zip(widths, self.aligner.power_alignment.align))

        alignments = []
        for (_, ref_token), (_, hyp_token), (_, align_token) in zip(s1_args, s2_args, align_args):
            if align_token == "C":
                op_type = OpType.MATCH
            if align_token == "S":
                op_type = OpType.SUBSTITUTE
            # NOTE: Insertions/deletions are reversed compared to our implementation.
            if align_token == "I":
                op_type = OpType.DELETE
            if align_token == "D":
                op_type = OpType.INSERT

            alignment = Alignment(
                op_type=op_type,
                ref=ref_token,
                hyp=hyp_token,
                left_compound=False,
                right_compound=False,
            )
            alignments.append(alignment)

        return alignments
