# this is a minimal version of cellrangers marginal tag assignment algorithm
# most of the code was directly copied from cellranger.feature.feature_assigner and edited as needed
import scipy
import logging

import numpy as np
import pandas as pd

from sklearn import mixture


COUNTS_DYNAMIC_RANGE = 50.0
MIN_COUNTS_PER_TAG = 1000
MAX_ALLOWED_PEARSON_CORRELATION = 0.5


class FeatureAssignments(object):  # pylint: disable=too-few-public-methods
    """A class for values returned by `FeatureAssigner.get_feature_assignments`.

    Replaces a dictionary with the following properties:

        {'bc_indices': np.array,
        'umi_counts': int,
        'is_contaminant': bool,
        'correlated_features': list}},

    where bc_indices is a np array that indexes into matrix.bcs,
    correlated_features is a list of feature_ids
    """

    def __init__(
        self,
        bc_indices: np.ndarray,
        umi_counts: int,
        is_contaminant: bool,
        correlated_features=None,
    ):
        self.bc_indices = bc_indices
        self.umi_counts = umi_counts
        self.is_contaminant = is_contaminant
        self.correlated_features = correlated_features

    def __getitem__(self, item):
        """This class replaces a bunch of untyped dictionaries.

        For backwards compatability we enable attr access this way.
        """
        return getattr(self, item)


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


def identify_contaminant_tags(
    tag_counts_matrix: scipy.sparse.csr_matrix,
    assignments: dict[str, FeatureAssignments],
    feature_names: np.ndarray,
    barcodes: np.ndarray,
    min_counts: int = MIN_COUNTS_PER_TAG,
    dynamic_range: float = COUNTS_DYNAMIC_RANGE,
    max_cor: float = MAX_ALLOWED_PEARSON_CORRELATION,
) -> dict[str, FeatureAssignments]:
    """
    Filter out tags that are likely contaminants based on total UMI counts across the
    cell barcodes and the correlation of counts per cell with tags with more UMI counts.

    Args:
        tag_counts_matrix (scipy.sparse.csr_matrix): tag count matrix containing umi counts per tag
        assignments (Dict[bytes, FeatureAssignments]): with keys being tag IDs
        feature_names (list[str]): list of feature names corresponding to tag_counts_matrix rows
        barcodes (list[str]): list of barcodes corresponding to tag_counts_matrix columns
        min_counts (int, optional): tags with total counts less than minimal will be
        contaminants. Defaults to MIN_COUNTS_PER_TAG.
        dynamic_range (float, optional): tags with total counts below this range from the
        top will be contaminants. Defaults to COUNTS_DYNAMIC_RANGE.
        max_cor (float, optional): tags with a Pearson correlation with a more abundant
        tag larger than it will be contaminants. Defaults to MAX_ALLOWED_PEARSON_CORRELATION.

    Returns:
        Dict[bytes, FeatureAssignments]
    """
    max_count = max(
        assignments[feature_name].umi_counts for feature_name in feature_names
    )
    contaminant_tags = set()

    all_features = sorted(assignments.keys())
    for feature_id in all_features:
        count = assignments[feature_id].umi_counts
        if count == 0:
            assignments.pop(feature_id, None)
            contaminant_tags.add(feature_id)
        elif count < min_counts or count * dynamic_range < max_count:
            assignments[feature_id].is_contaminant = True
            contaminant_tags.add(feature_id)

    all_features = sorted(assignments.keys())

    # correlation check
    df = pd.DataFrame(
        tag_counts_matrix.toarray(),
        index=feature_names,
        columns=barcodes,
    ).T
    df = df.loc[:, all_features]
    cor_df = df.corr()
    for i in range(len(all_features) - 1):
        for j in range(i + 1, len(all_features)):
            this_cor = cor_df.loc[all_features[i], all_features[j]]
            if np.isnan(this_cor) or this_cor < max_cor:
                continue
            if assignments[all_features[i]].umi_counts < (
                assignments[all_features[j]].umi_counts
            ):
                assignments[all_features[i]].is_contaminant = True
                assignments[all_features[i]].correlated_features.append(all_features[j])
            else:
                assignments[all_features[j]].is_contaminant = True
                assignments[all_features[j]].correlated_features.append(all_features[i])

    return assignments


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


def create_tag_assignments_df(
        assignments: dict[str, FeatureAssignments],
        feature_names: np.ndarray,
        barcodes: np.ndarray
) -> pd.DataFrame:
    """Create a feature assignment matrix

    rows are features and columns are barcodes.
    In each cell, the value is 1 if barcode was assigned this feature, otherwise 0.
    """
    feature_assignments_df = pd.DataFrame()
    for f_id in feature_names:
        asst = assignments[f_id]
        if asst.is_contaminant:  # only relevant for multiplexing library
            continue

        mask = np.zeros(len(barcodes))
        if len(asst.bc_indices) > 0:
            # Put 1 for barcodes that were assigned to this feature. Basically, if
            # bc_indices = [1, 4], go from  [0, 0, 0, 0, 0] -> [0, 1, 0, 0, 1]
            np.put(mask, asst.bc_indices, 1)
        # instead of creating a dense df and then putting it in a sparse matrix,
        # better to create sparse arrays and put those into sparse df
        sparse_mask = pd.arrays.SparseArray(mask, dtype="uint8")
        feature_assignments_df[f_id] = sparse_mask
        del mask

    feature_assignments_df.index = barcodes
    feature_assignments_df = feature_assignments_df.T

    return feature_assignments_df


def get_features_per_cell_table(
        tag_counts_matrix: scipy.sparse.csr_matrix,
        feature_assignments_df: pd.DataFrame,
        feature_names: np.ndarray,
        sep: str = "|"
) -> pd.DataFrame:
    """Returns a dataframe of features assigned, with metadata.

    Returns a Pandas dataframe, indexed by bc, that provides a list of features assigned and
    metadata. Columns are ['num_features', 'feature_call', 'num_umis']
    Creates the "tag_calls_per_cell" output for Multiplexing Capture library
    """

    columns = ['num_features', 'feature_call', 'num_umis']
    features_per_cell_table = pd.DataFrame(columns=columns)

    # apparently itertuples() is faster than iterrows()
    for i, row in enumerate(feature_assignments_df.T.itertuples()):
        cell = row[0]
        calls = np.asarray(np.array(row[1:]) > 0).nonzero()
        calls = feature_assignments_df.index[calls].values

        num_features = len(calls)
        # TODO: Make a separate PR to include these cells
        if num_features == 0:
            continue

        # These three lines represent a fast way of populating the UMIS
        features = sep.join(calls)
        bc_data = np.ravel(
            tag_counts_matrix[i].todense()
        )  # get data for a single bc, densify it, then flatten it
        feature_names = list(feature_names)
        umis = sep.join(
            [str(bc_data[feature_names.index(f_id)]) for f_id in calls]
        )

        features_per_cell_table.loc[cell] = (num_features, features, umis)

    features_per_cell_table.index.name = 'cell_barcode'
    features_per_cell_table.sort_values(by=['feature_call'], inplace=True, kind="mergesort")
    return features_per_cell_table


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
