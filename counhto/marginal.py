# this is a minimal version of cellrangers marginal tag assignment algorithm
# most of the code was directly copied from cellranger.feature.feature_assigner and edited as needed
import scipy
import logging

import numpy as np

from sklearn import mixture
from .utils import (
    FeatureAssignments,
    identify_contaminant_tags,
    get_features_per_cell_table,
    create_tag_assignments_df
)


def call_presence_with_gmm_ab(umi_counts: np.ndarray, n_components: int = 2) -> np.ndarray:
    """ Given the UMI counts for a specific antibody, separate signal from background """

    if np.max(umi_counts) == 0:
        # there are no UMIs, each barcode has 0 count
        return np.repeat(False, len(umi_counts))

    # Turn the numpy array into 2D log scale array
    umi_counts = np.reshape(umi_counts, (len(umi_counts), 1))
    logAb = np.log10(umi_counts + 1.0)

    # Initialize and fit the gaussian Mixture Model
    gmm = mixture.GaussianMixture(n_components, n_init=10, covariance_type="tied", random_state=0)
    gmm.fit(logAb)

    # Calculate the threshold
    umi_posterior = gmm.predict_proba(logAb)
    high_umi_component = np.argmax(gmm.means_)

    in_high_umi_component = umi_posterior[:, high_umi_component] > 0.5
    # umi_threshold = np.power(10, np.min(logAb[in_high_umi_component]))

    return in_high_umi_component


def get_tag_assignments(
        tag_counts_matrix: scipy.sparse.csr_matrix,
        feature_names: np.ndarray,
        barcodes: np.ndarray
) -> dict[str, FeatureAssignments]:
    assignments: dict[str, np.array] = {}
    for i, feature_name in enumerate(feature_names):
        umi_counts = tag_counts_matrix[:, i].toarray().flatten()
        in_high_umi_component = call_presence_with_gmm_ab(umi_counts)
        assignments[feature_name] = FeatureAssignments(
            np.flatnonzero(np.array(in_high_umi_component)), sum(umi_counts), False, []
        )

    assignments = identify_contaminant_tags(
        tag_counts_matrix.T,
        assignments,
        feature_names,
        barcodes
    )

    return assignments


def get_marginal_tag_assignment(
        tag_counts_matrix: scipy.sparse.csr_matrix,
        feature_names: np.ndarray,
        barcodes: np.ndarray
):
    logging.info('computing marginal tag assignments')
    assignments = get_tag_assignments(
        tag_counts_matrix,
        feature_names,
        barcodes
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
