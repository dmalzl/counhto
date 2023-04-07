# this is a minimal version of cellrangers jibes tag assignment algorithm
# most of the code was directly copied from cellranger.feature.jibes_tag_assigner and edited as needed
import time
import scipy
import logging

import pandas as pd
import numpy as np

from enum import Enum
from .marginal import (
    FeatureAssignments,
    create_tag_assignments_df,
    get_features_per_cell_table
)
from .jibes_data import JibesData
from .jibes_py import (
    N_G,
    MULTIPLETS_FACTOR_NAME,
    BLANK_FACTOR_NAME,
    UNASSIGNED_FACTOR_NAME,
    ASSIGNMENT_COL_NAME,
    JIBES_MIN_CONFIDENCE
)
from .jibes_py import JibesEMPy as JibesEM
from .jibes_py import (
    initial_params_from_tag_calls_df,
    get_assignment_df,
    get_cols_associated_with_assignments
)


class FitStatus(Enum):
    NotStarted = 0
    Finished = 1
    FailedToConverge = 2


def get_tag_assignments(
        tag_counts_matrix: scipy.sparse.csr_matrix,
        feature_names: list[str],
        barcodes: list[str],
        features_per_cell_table: pd.DataFrame,
        n_gems: int = N_G,
        confidence: float = 0.9
):
    """
    Get Tag assignments for this dataset after running fitting.

    There is some nuance in how these values are computed for the JIBES caller as oppposed
    to the older marginal ones.  In particular, this function normally lists all the barcodes
    that are "above" background for a given tag.  In the JIBES caller, instead of above/below background for
    each tag, we estimate which latent state (either a singlet or multiplet) a barcode might
    be in and assign a probability value to every such state.

    To transform the joint inference back into the marginal inference, we do the following
    operations:

    - If a barcode is "Blanks" it is not assigned to any FeatureAssignment
    - If a barcode is "Unassigned", it is not assigned to any FeatureAssignment due to lack of confidence
    - If a barcode is a Singlet, it is assigned to to the FeatureAssignment for that tag.
    - If a barcode is a multiplet, we find the most likely multiplet state and assign the barcode to all tags > 0 for that state.

    Note that in JIBES a barcode is assigned as a multiplet if the summed probability over
    all multiplet latent states is greater than any singlet state, so the `most likely multiplet`
    state, might not be the most likely state.

    :return: A dictionary of {tag_name:[FeatureAssignmentObjects]
    """
    data = JibesData(tag_counts_matrix, feature_names, barcodes)
    prelim_assignments = features_per_cell_table.reset_index().drop(
        columns=['num_features', 'num_umis']
    )
    init_data = data.subset_to_barcodes(
        prelim_assignments['cell_barcode'].astype("S")
    )
    starting_model = initial_params_from_tag_calls_df(
        prelim_assignments, init_data, n_gems=n_gems
    )

    fitter = JibesEM(data, starting_model, n_gems=n_gems)

    logging.info('start fitting Jibes model')
    fitter.perform_EM(abs_tol=0.1)
    if not fitter.converged:
        logging.info('fitter not converged. results might be suboptimal.')

    else:
        logging.info('converged.')

    cmo_cnts = fitter.data.get_df_from_data()
    jibes_assignments = get_assignment_df(fitter, confidence=confidence)
    col_names = fitter.data.column_names
    name_to_index = {name: i for i, name in enumerate(col_names)}
    indices = [[] for _ in range(len(col_names))]
    multiplet_latent_state_cols = get_cols_associated_with_assignments(fitter)[
        MULTIPLETS_FACTOR_NAME
    ]
    # Posterior is dimenstions n_bcs x n_latent_states
    multiplet_posterior = fitter.posterior[:, multiplet_latent_state_cols]
    # X is dimensions n_latent_states x n_tags
    mutiplet_latent_states = fitter.X[multiplet_latent_state_cols, 1:]

    def get_most_likely_multiplet_state(row_index):
        max_prob_index = np.argmax(multiplet_posterior[row_index, :])
        latent_states = mutiplet_latent_states[max_prob_index, :]
        return latent_states

    non_tag_assignments = {
        BLANK_FACTOR_NAME: [],
        UNASSIGNED_FACTOR_NAME: [],
    }
    for i in range(cmo_cnts.shape[0]):
        asn = jibes_assignments[ASSIGNMENT_COL_NAME][i]
        if asn in non_tag_assignments:
            non_tag_assignments[asn].append(i)
            continue
        elif asn == MULTIPLETS_FACTOR_NAME:
            most_probable_multiple_state = get_most_likely_multiplet_state(i)
            for k, count in enumerate(most_probable_multiple_state):
                if count > 0.0:
                    indices[k].append(i)
        else:  # Singlet
            col = name_to_index[asn]
            indices[col].append(i)

    assignments = {}
    for j, c_name in enumerate(col_names):
        umi_counts = tag_counts_matrix[:, j].toarray().flatten()
        assignments[c_name] = FeatureAssignments(indices[j], umi_counts.sum(), False, [], )

    # TODO: This is handled upstream of processing currently, so we dont' replicate this
    # logic here
    # assignments = self.identify_contaminant_tags()
    return assignments


def get_jibes_tag_assignment(
        tag_counts_matrix: scipy.sparse.csr_matrix,
        feature_names: list[str],
        barcodes: list[str],
        marginal_features_per_cell_table: pd.DataFrame,
        n_gems: int = N_G,
        confidence: float = JIBES_MIN_CONFIDENCE
):
    logging.info('compute Jibes tag assignments')
    assignments = get_tag_assignments(
        tag_counts_matrix,
        feature_names,
        barcodes,
        marginal_features_per_cell_table,
        n_gems,
        confidence
    )
    feature_assignments_df = create_tag_assignments_df(
        assignments,
        feature_names,
        barcodes
    )
    features_per_cell_table = get_features_per_cell_table(
        tag_counts_matrix,
        feature_assignments_df,
        feature_names
    )

    return features_per_cell_table

