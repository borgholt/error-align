# -*- coding: utf-8 -*-
import random

import numpy as np
from rapidfuzz.distance import Levenshtein

from error_align.utils import DELIMITERS, OpType, OP_TYPE_COMBO_MAP, OP_TYPE_COMBO_MAP_INV


def _get_levenshtein_values(ref_token: str, hyp_token: str):
    """Compute the Levenshtein values for deletion, insertion, and diagonal (substitution or match).

    Args:
        ref_token (str): The reference token.
        hyp_token (str): The hypothesis token.

    Returns:
        tuple: A tuple containing the deletion cost, insertion cost, and diagonal cost.
    """
    if hyp_token == ref_token:
        diag_cost = 0
    else:
        diag_cost = 1

    return 1, 1, diag_cost


def _get_optimal_word_alignment_values(ref_token: str, hyp_token: str):
    """Compute the optimal word alignment values for deletion, insertion, and diagonal (substitution or match).

    Args:
        ref_token (str): The reference token.
        hyp_token (str): The hypothesis token.

    Returns:
        tuple: A tuple containing the deletion cost, insertion cost, and diagonal cost.
    """
    if hyp_token == ref_token:
        diag_cost = 0
    else:
        diag_cost = Levenshtein.normalized_distance(ref_token, hyp_token)
        assert 0 < diag_cost <= 1, "Diagonal cost should be between 0 and 1."
        # diag_cost = Levenshtein.distance(ref_token, hyp_token) / len(ref_token)

    return 1, 1, diag_cost


def _get_error_align_values(ref_token: str, hyp_token: str):
    """Compute the error alignment values for deletion, insertion, and diagonal (substitution or match).

    Args:
        ref_token (str): The reference token.
        hyp_token (str): The hypothesis token.

    Returns:
        tuple: A tuple containing the deletion cost, insertion cost, and diagonal cost.
    """
    if hyp_token == ref_token:
        diag_cost = 0
    elif hyp_token in DELIMITERS or ref_token in DELIMITERS:
        diag_cost = 3  # NOTE: Will never be chosen as insert + delete (= 2) is equivalent and cheaper.
    else:
        diag_cost = 2

    return 1, 1, diag_cost


def compute_distance_matrix(
    ref: str | list[str],
    hyp: str | list[str],
    score_func: callable,
    backtrace: bool = False,
    dtype: type = int,
):
    """
    Compute the edit distance score matrix between two sequences x (hyp) and y (ref).

    Args:
        ref (str): The reference sequence/transcript.
        hyp (str): The hypothesis sequence/transcript.
        score_func (callable): A function that takes two tokens (ref_token, hyp_token) and returns a tuple
            of (deletion_cost, insertion_cost, diagonal_cost).
        backtrace (bool): Whether to compute the backtrace matrix.

    Returns:
        np.ndarray: The score matrix.
        np.ndarray: The backtrace matrix, if backtrace=True.
    """

    # Create empty score matrix of zeros and initialize first row and column
    hyp_dim, ref_dim = len(hyp) + 1, len(ref) + 1
    D = np.zeros((hyp_dim, ref_dim), dtype=dtype)
    D[0, :] = np.arange(ref_dim)
    D[:, 0] = np.arange(hyp_dim)

    # Create backtrace matrix and operation combination map and initialize first row and column
    # Each operation combination is dynamically assigned a unique integer
    if backtrace:
        B = np.zeros((hyp_dim, ref_dim), dtype=int)
        B[0, 0] = OP_TYPE_COMBO_MAP_INV[(OpType.MATCH,)]  # start tokens always match
        B[0, 1:] = OP_TYPE_COMBO_MAP_INV[(OpType.INSERT,)]  # implies horisontal step
        B[1:, 0] = OP_TYPE_COMBO_MAP_INV[(OpType.DELETE,)]  # implies vertical step

    # Fill in the score and backtrace matrix
    for j in range(1, ref_dim):
        for i in range(1, hyp_dim):

            del_cost, ins_cost, diag_cost = score_func(ref[j - 1], hyp[i - 1])

            # Compute the new value
            del_val = D[i - 1, j] + del_cost
            ins_val = D[i, j - 1] + ins_cost
            diag_val = D[i - 1, j - 1] + diag_cost
            new_val = min(del_val, ins_val, diag_val)
            D[i, j] = new_val

            # Track possible operations (note that the order of operations matters)
            if backtrace:
                pos_ops = tuple()
                if diag_val == new_val and diag_cost == 0:
                    pos_ops += (OpType.MATCH,)
                if ins_val == new_val:
                    pos_ops += (OpType.INSERT,)
                if del_val == new_val:
                    pos_ops += (OpType.DELETE,)
                if diag_val == new_val and diag_cost > 0:
                    pos_ops += (OpType.SUBSTITUTE,)
                B[i, j] = OP_TYPE_COMBO_MAP_INV[pos_ops]

    if backtrace:
        return D, B
    else:
        return D


def compute_levenshtein_distance_matrix(ref: str | list[str], hyp: str | list[str], backtrace: bool = False):
    """
    Compute the Levenshtein distance matrix between two sequences.

    Args:
        ref (str): The reference sequence/transcript.
        hyp (str): The hypothesis sequence/transcript.
        backtrace (bool): Whether to compute the backtrace matrix.

    Returns:
        np.ndarray: The score matrix.
        np.ndarray: The backtrace matrix, if backtrace=True.
    """
    return compute_distance_matrix(ref, hyp, _get_levenshtein_values, backtrace)


def compute_error_align_distance_matrix(ref: str | list[str], hyp: str | list[str], backtrace: bool = False):
    """
    Compute the error alignment distance matrix between two sequences.

    Args:
        ref (str): The reference sequence/transcript.
        hyp (str): The hypothesis sequence/transcript.
        backtrace (bool): Whether to compute the backtrace matrix.

    Returns:
        np.ndarray: The score matrix.
        np.ndarray: The backtrace matrix, if backtrace=True.
    """
    return compute_distance_matrix(ref, hyp, _get_error_align_values, backtrace)


def compute_optimal_word_alignment_distance_matrix(ref: list[str], hyp: list[str], backtrace: bool = False):
    """
    Compute the optimal word alignment distance matrix between two sequences.

    Args:
        ref (list[str]): The reference sequence/transcript.
        hyp (list[str]): The hypothesis sequence/transcript.
        backtrace (bool): Whether to compute the backtrace matrix.

    Returns:
        np.ndarray: The score matrix.
        np.ndarray: The backtrace matrix, if backtrace=True.
    """
    return compute_distance_matrix(ref, hyp, _get_optimal_word_alignment_values, backtrace, dtype=float)


def get_optimal_alignment_trace(B, sample=False):
    """
    Find the edit operations from the backtrace matrix B.

    Args:
        B (np.ndarray): The backtrace matrix.
        ref (list[str]|str): The reference sequence/transcript.
        hyp (list[str]|str): The hypothesis sequence/transcript.
        sample (bool): Whether to sample from the operation combinations or just use the first op deterministically.

    Returns:
        list: The list of operations.
    """
    i, j = B.shape[0] - 1, B.shape[1] - 1

    alignment_trace = []
    while i > 0 or j > 0:

        ops = OP_TYPE_COMBO_MAP[B[i, j]]

        if sample:
            op_type = random.choice(ops)
        else:
            op_type = ops[0]

        alignment_trace.append((op_type, (i - 1, j - 1)))

        if op_type == OpType.MATCH:
            i, j = i - 1, j - 1
        elif op_type == OpType.INSERT:
            j = j - 1
        elif op_type == OpType.DELETE:
            i = i - 1
        elif op_type == OpType.SUBSTITUTE:
            i, j = i - 1, j - 1

    return alignment_trace[::-1]
