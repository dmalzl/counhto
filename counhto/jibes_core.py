# Copyright (c) 2020 10X Genomics, Inc. All rights reserved.
"""
Python implementation of classes also implemented in Rust.  This was the original implementation
before these classes were migrated to PyO3.  They are kept around to enable easier
prototyping of model changes and because they are occasionally passed as pickles.
"""
from __future__ import absolute_import, print_function, division
from six import ensure_str

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm, poisson

from .jibes_data import JibesData
from .combinatorics import multinomial_comb, generate_all_multiplets
from scipy import optimize
from typing import Union

DEFAULT_BLANK_PROB = 0.04
_MAX_K_LETS_TO_CONSIDER = 3
CORR_FACTOR = 1.54
NUM_TOTAL_TAGS = 14
N_G = 95000
G19_N_GEMS = {"MT": N_G, "LT": 9500, "HT": 190000}
JIBES_MIN_CONFIDENCE = 0.9

MULTIPLETS_FACTOR_NAME = "Multiplet"
UNASSIGNED_FACTOR_NAME = "Unassigned"
BLANK_FACTOR_NAME = "Blanks"
ASSIGNMENT_COL_NAME = "Assignment"
ASSIGNMENT_PROB_COL_NAME = "Assignment_Probability"
# pylint: disable=invalid-name


# copied from tenkit.stats
def robust_divide(a, b):
    # type: (Union[float, int, str, bytes], Union[float, int, str, bytes]) -> float
    """Handles 0 division and conversion to floats automatically"""
    a = float(a)
    b = float(b)
    if b == 0:
        return float("NaN")
    else:
        return a / b


# copied from cellranger.feature.feature_assigner
def get_poisson_multiplets_fractions(loading_rate, max_klet: int = NUM_TOTAL_TAGS):
    """Returns the fractions of GEMs for each n-let.

    Given a loading_rate and the total number of tags,
    returns fractions of GEMs that would be 1-let, 2-let ..., n-let
    """
    rv = poisson(loading_rate)
    k_lets = range(1, max_klet + 1)
    return rv.pmf(k_lets)


def get_multiplet_counts_unrounded(obs_cells: float, n_gems: int = N_G) -> np.ndarray:
    """Returns unrounded counts for multiplets 1 and above."""
    num_loaded_cells = CORR_FACTOR * obs_cells
    # estimate the numnber of cells loaded from the number of recovered cells
    loading_rate = float(num_loaded_cells) / n_gems  # Poisson loading rate
    # expected counts of i-lets just due to Poisson loading
    poisson_fracs = get_poisson_multiplets_fractions(loading_rate)
    poisson_counts = poisson_fracs * n_gems / CORR_FACTOR
    return poisson_counts


def get_multiplet_counts(obs_cells: float, n_gems: int = N_G) -> np.ndarray:
    return np.round(get_multiplet_counts_unrounded(obs_cells, n_gems))


def calculate_expected_total_cells(obs_barcodes: int, n_gems: int = N_G) -> float:
    """Solve for the observed cell counts.

    Given an observed number of cell barcodes, try to solve for
    the number of likely observed cells in the original mixture by estimating the number
    of different k-lets implied by that total number of barcodes.
    """
    # TODO: Quick hack to make multiplexing work without passing through a
    # `throughput` parameter, will need to remove the throughput parameter later.
    # Needed to assess 60K sample on G19
    if obs_barcodes > 61688:
        n_gems = G19_N_GEMS["HT"]
    to_minimize = lambda x: np.power(
        obs_barcodes - np.sum(get_multiplet_counts_unrounded(x, n_gems)), 2.0
    )

    z = optimize.minimize(to_minimize, x0=obs_barcodes * 1.1)
    if not z.success:
        raise RuntimeError("Could not estimate number of cells based on number of barcodes")
    if z.fun > 2:
        raise RuntimeError("Could not converge on an accurate cell estimate.")
    return z.x[0]


# copied from cellranger.analysis.jibes
def create_initial_parameters(data, assignments, n_gems=N_G):
    """
    Create initial conditions based on initial assignments

    Args:
        data:  a JibesData instance
        assignments: an ndarray ndarray where each element is a string matching the column name of data
    (or irrelevant) create initial conditions for the jibes model to fit

    Returns: A JibesModel with initial conditions

    """
    # pylint: disable=too-many-locals
    k = len(data.column_names)
    foreground_means = np.zeros(k)
    std_devs = np.zeros(k)
    backgrounds = np.zeros(k)
    bad_indices = set()
    singletons = np.isin(assignments, data.column_names)
    for i, name in enumerate(data.column_names):
        vals = assignments == name
        if np.sum(vals) < 2:
            foreground_means[i] = np.nan
            backgrounds[i] = np.nan
            std_devs[i] = np.nan
            bad_indices.add(i)
        else:
            # TODO: Assuming only one library in the data
            tmp = assignments != name
            other_singletons = tmp & singletons
            if np.sum(other_singletons) > 0:
                backgrounds[i] = data.counts[other_singletons, i].mean()
            else:
                backgrounds[i] = data.counts[:, i].mean()
            foreground_counts = data.counts[vals, i].squeeze()
            bg = backgrounds[i]
            foreground_means[i] = np.maximum(0.6 + bg, np.mean(foreground_counts)) - bg
            std_devs[i] = foreground_counts.std()
    # If the marginal caller goes goofy, sometimes we have no singlet calls for a tag,
    # and we'll fill in with the average of other values
    if len(bad_indices) > 0:
        print("Bad indices")
        bad_indices = list(bad_indices)
        if len(bad_indices) == len(data.column_names):
            # TODO: Need Better Initial Conditions
            foreground_means[bad_indices] = 1.0
            backgrounds[bad_indices] = 0.5
            std_devs[bad_indices] = 0.3
            # raise RuntimeError("None of the Tags had singleton calls in the earlier data.")
        else:
            good = list(x for x in range(len(data.column_names)) if x not in bad_indices)
            fm = np.mean(foreground_means[good])
            bm = np.mean(backgrounds[good])
            vm = np.mean(std_devs[good])
            foreground_means[bad_indices] = fm
            backgrounds[bad_indices] = bm
            std_devs[bad_indices] = vm
    # TODO: Improve this edge case handling
    zero_std_devs = np.where(std_devs == 0.0)[0]
    if len(zero_std_devs) > 0:
        print("Edge case with no variance in initial calls detected!!")
        std_devs[zero_std_devs] = 0.2
    model = JibesModelPy(backgrounds, foreground_means, std_devs, n_gems=n_gems)
    return model


def _order_initial_parameters(assignments, data):
    """
    Helper function to make sure the barcodes in two dataframes used for initialization
    are in the same order.

    :param assignments:
    :param data:
    :return:
    """
    bc_to_index = {x: i for i, x in enumerate(assignments['cell_barcode'])}
    indices = np.fromiter(
        (bc_to_index[x] for x in data.barcodes), dtype=np.int32, count=len(data.barcodes)
    )
    return assignments.take(indices).reset_index(drop=True)


def initial_params_from_tag_calls_df(assignments, data, n_gems=N_G):
    """
    Given a DataFrame loaded from tag_calls_per_cell_fn created by CALL_TAGS_PD, make initial parameter guesses to fit a
    JIBES model with by utilizing the earlier cell calls
    :param tag_calls_per_cell: Pandas DataFrame produced from loading a tag_calls_per_cell file
    :param data: A JibesData object
    :return: A JibesModel with initial conditions
    """
    # pylint: disable=too-many-locals
    assert isinstance(data, JibesData)
    assert assignments.shape[0] == len(data.barcodes)
    assignments = _order_initial_parameters(assignments, data)
    return create_initial_parameters(
        data, assignments['feature_call'].values, n_gems=n_gems
    )


def get_cols_associated_with_assignments(fit):
    """
    From a fit model, determine which cols are associated with the
    blank, singlet (or doublets of same type) and multiplet latent states.
    :param fit:
    :return: a dictionary that goes from assignment (numeric for singlets) ->row_indices
    """
    isinstance(fit, JibesEMPy)
    X = fit.X[:, 1:]
    row_sums = np.sum(X, axis=1)
    all_indices = set(range(1, X.shape[0]))
    matches = {}
    for k in range(fit.k):
        all_tag_rows = [i for i in range(1, X.shape[0]) if X[i, k] == row_sums[i]]
        matches[k] = all_tag_rows
        for x in all_tag_rows:
            all_indices.discard(x)
    # Now the zero category, silly cut/paste, it's the first row
    all_tag_rows = [i for i in range(X.shape[0]) if row_sums[i] == 0]
    matches[BLANK_FACTOR_NAME] = all_tag_rows
    for x in all_tag_rows:
        all_indices.discard(x)
    matches[MULTIPLETS_FACTOR_NAME] = list(all_indices)
    return matches


def _enforce_min_confidence(
    candidate_asst,
    asst_probability,
    confidence=JIBES_MIN_CONFIDENCE,
    unassigned_name=UNASSIGNED_FACTOR_NAME,
):
    if asst_probability < confidence:
        return unassigned_name
    return candidate_asst


def get_assignment_df(fit, is_jibes_multi_genome=False, confidence=JIBES_MIN_CONFIDENCE):
    """
    Returns a Pandas dataframe where each row is a barcode, and each column is either a feature ID, one of
    [barcodes, Multiplet, Blanks, Assignment, Assignment_Probability].
    Each value is the posterior probability for that barcode to be in that state.
    Note that the df is not indexed by barcodes, that info is in the barcodes column.
    Also, Assignemnt is either a feature ID (singlet) or Blanks (no CMO state) or
    Unassigned (no single state is likely enough to exceed the minimum confidence required).
    Assignment_Probability contains the probability of the most likely state, regardless of whether or not it meets the minimum threshold).
    """
    assert isinstance(fit, JibesEMPy)
    df = fit.data.get_df_from_data()
    matches = get_cols_associated_with_assignments(fit)
    all_names = []
    posterior = fit.posterior  # only get this once as it's a copy operation from the Rust side
    for i, col in enumerate(fit.data.column_names):
        indices = matches[i]
        name = col
        all_names.append(name)
        probs = posterior[:, indices].sum(axis=1)
        df[name] = probs

    indices = matches[MULTIPLETS_FACTOR_NAME]
    all_names.append(MULTIPLETS_FACTOR_NAME)
    probs = posterior[:, indices].sum(axis=1)
    df[MULTIPLETS_FACTOR_NAME] = probs

    indices = matches[BLANK_FACTOR_NAME]
    all_names.append(BLANK_FACTOR_NAME)
    probs = posterior[:, indices].sum(axis=1)
    df[BLANK_FACTOR_NAME] = probs

    df[ASSIGNMENT_COL_NAME] = df[all_names].idxmax(axis=1)
    df[ASSIGNMENT_PROB_COL_NAME] = df[all_names].max(axis=1)

    if is_jibes_multi_genome:
        return df

    df[ASSIGNMENT_COL_NAME] = df.apply(
        lambda row: _enforce_min_confidence(
            row[ASSIGNMENT_COL_NAME], row[ASSIGNMENT_PROB_COL_NAME], confidence=confidence,
        ),
        axis=1,
    )
    df.index.name = "Row"
    return df


class JibesModelPy:
    """ A class that stores parameters and can simulate from a JIBES model"""

    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        backgrounds,
        foregrounds,
        std_devs,
        frequencies=None,
        blank_prob=DEFAULT_BLANK_PROB,
        n_gems=N_G,
    ):
        num_tags = len(backgrounds)
        assert len(backgrounds) == num_tags
        assert len(foregrounds) == num_tags
        assert len(std_devs) == num_tags
        if frequencies:
            assert len(frequencies) == num_tags
            self.frequencies = frequencies
        else:
            self.frequencies = np.repeat(robust_divide(1, num_tags), num_tags)
        self.blank_freq = blank_prob
        self.num_tags = num_tags
        self.background = np.array(backgrounds)
        self.foreground = np.array(foregrounds)
        self.B = np.vstack((self.background, np.diag(self.foreground)))
        self.std_devs = np.array(std_devs)
        self.n_gems = n_gems
        if np.sum(np.isnan(self.B)) != 0:
            raise ValueError("NaN Detected in Coefficient Matrix")
        if np.sum(np.isnan(self.std_devs)):
            raise ValueError("NaN Detected in Std. Devs.")

    def simulate(self, N):
        """
        Simulate a set of observations given an input number of "observed" cells
        :param N: How many cells would be recovered across all k-lets?
        :return: A tuple with matrix Y, giving the observed counts, and a matrix X, with the
                 simulated latent states
        """
        # TODO: Blank state not yet implemented
        cnts = [int(x) for x in get_multiplet_counts(N, self.n_gems)][: self.num_tags]
        total = np.sum(cnts)
        x = np.zeros((total, self.num_tags + 1))
        x[:, 0] = np.ones(total)
        cur_row = 0
        for index, cnt in enumerate(cnts):
            to_sample = (index + 1) * cnt
            samps = np.random.choice(self.num_tags, to_sample)
            start_row = cur_row
            for i in range(index + 1):
                cur_row = start_row
                for j in range(cnt):
                    x[cur_row, 1 + samps[j + i * j]] += 1
                    cur_row += 1
        means = np.matmul(x, self.B)
        Y = np.zeros((total, self.num_tags))
        for k in range(self.num_tags):
            Y[:, k] = np.random.normal(means[:, k], self.std_devs[k])
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        assert self.num_tags < len(alphabet)
        data = JibesData(Y, list(alphabet[: self.num_tags]), None)
        return (data, x)

    def foreground_as_vector(self):
        """ Return the diagonal matrix as a vector of the diagnoals instead """
        return np.diag(self.foreground)


class JibesEMPy:  # pylint: disable=too-many-instance-attributes
    """ Class to perform EM fitting of JIBES model """

    def __init__(
        self,
        data,
        model,
        optimize_pop_freqs=False,
        max_k_lets=_MAX_K_LETS_TO_CONSIDER,
        n_gems=N_G,
    ):
        """
        Fit a Joint Inference by Exploiting Stoichiometry
        :param data:
        :param model:
        :param optimize_pop_freqs: Optimize tag populations (otherwise all equal)
        :param max_k_lets: Maximum k-lets to consider
        :param n_gems: Number of GEMs used
        """
        assert isinstance(model, JibesModelPy)
        assert isinstance(data, JibesData)
        self.optimize_pop_freqs = optimize_pop_freqs
        n = data.counts.shape[0]
        self.estimated_cells = calculate_expected_total_cells(n, n_gems=n_gems)
        self.max_k_let_setting = max_k_lets
        self.n_gems = n_gems
        exp_cnts = get_multiplet_counts(self.estimated_cells, n_gems=n_gems)
        # Only integrate up to a multiplet number
        # with expected count >= 1, as multiplets of higher integer values lead to
        # combinatorial explosions
        max_multiplets = np.max(np.nonzero(exp_cnts)) + 1
        max_multiplets = max(max_multiplets, 2)
        if max_multiplets > self.max_k_let_setting:
            # For this version of the algorithm, to save memory we'll have a "catch-all" category
            # that has all the states, and limit the set considered to _MAX_K_LETS_TO_CONSIDER,
            # from scanning data, it doesn't appear the doublets/triplets of one cell type are
            # that different from the singlets
            # TODO: Verify this assumption and add tests
            print(
                "Limiting Latent States to K-lets of {} with a blank state".format(
                    self.max_k_let_setting
                )
            )
            self.k_let_limited = True
            max_multiplets = self.max_k_let_setting
        else:
            self.k_let_limited = False
        latent_states = generate_all_multiplets(model.num_tags, max_multiplets, self.k_let_limited)
        self.max_modeled_k_let = max(model.num_tags, max_multiplets)
        self.latent_states = np.array(latent_states)
        print("Total Latent States = " + str(len(self.latent_states)))
        print("Observed Barcodes = {}".format(n))
        print("Inferred Cells = {}".format(self.estimated_cells))
        self.data = data
        self.model = model
        ones = np.ones((self.z, 1))
        self.X = np.hstack((ones, self.latent_states))
        # pylint: disable=invalid-name
        self.data_full, self.X_full = self._reshape_for_regression()
        self.posterior = None
        self.LL = np.NINF
        self.converged = False
        self.iterations = 0

    @property
    def col_names(self):
        """ Returns the names of the vector """
        return self.data.column_names

    @property
    def n(self):
        """ Number of observed barcodes"""
        return self.data.counts.shape[0]

    @property
    def z(self):
        """ Number of latent states in the model """
        return self.latent_states.shape[0]

    @property
    def k(self):
        """ Number of observations per data point"""
        return self.model.num_tags

    def _calculate_latent_state_weights(self):
        """Based on our poisson expectation, calculate the probability
        of each type of singlet and each type of multiplets."""
        # cnts here starts at 1 cell, the 0 group is not included
        cnts = get_multiplet_counts_unrounded(self.estimated_cells, n_gems=self.n_gems)[
            : self.max_modeled_k_let
        ]
        p_k_let = cnts / np.sum(cnts)
        if self.k_let_limited:
            # We mush all the probability for the higher states into the last state that we want to account for all that data
            # Note no +1 here as it's 1 based array for multiple
            p_k_let[-1] = np.sum(p_k_let[self.max_k_let_setting :])
        x = self.X[:, 1:]
        klet = x.sum(axis=1).astype(np.int32)
        state = p_k_let[klet[1:] - 1]
        state = np.log(state)
        pis = np.log(self.model.frequencies)
        z = x.shape[0]
        comb_probs = np.zeros(z)
        comb_probs[0] = np.log(self.model.blank_freq)
        log_not_blank = np.log(1.0 - self.model.blank_freq)
        for zi in range(1, z):
            relevant_ps = np.nonzero(x[zi, :])
            cnts = x[zi, relevant_ps].flatten().astype(np.int64)
            ps = pis[relevant_ps]
            p = np.sum(cnts * ps)
            # TODO: Investigate precision issues when multiplier is large
            multiplier = float(multinomial_comb(cnts))
            comb_probs[zi] = p + np.log(multiplier) + state[zi - 1] + log_not_blank
        return comb_probs

    def _calculate_posterior_by_state(self):
        # Critical bit for the E step
        mu = np.matmul(self.X, self.model.B)
        z_prior = self._calculate_latent_state_weights()
        ll_posterior = np.tile(z_prior, (self.n, 1))
        for i in range(self.n):
            ll_posterior[i, :] += norm.logpdf(self.data.counts[i, :], mu, self.model.std_devs).sum(
                axis=1
            )
        ll_max = np.max(ll_posterior, axis=1)
        # TODO: There must be a better way than transposing here
        ll_posterior_altered = ll_posterior.T - ll_max
        posterior = np.exp(ll_posterior_altered.T)
        marginal = posterior.sum(axis=1, keepdims=True)
        self.posterior = posterior / marginal
        self.LL = np.log(marginal).sum() + ll_max.sum()

    def _maximize_parameters(self):
        """ Maximize the parameters by weighted regression"""
        # The M-step
        W = self.posterior.flatten()
        new_std_devs = np.zeros(self.k)
        new_foregrounds = np.zeros(self.k)
        new_backgrounds = np.zeros(self.k)
        for k in range(self.k):
            res_wls = sm.WLS(self.data_full[:, k], self.X_full[:, [0, k + 1]], weights=W).fit()
            # pylint: disable=no-member
            residuals = res_wls.fittedvalues - np.squeeze(np.asarray(self.data_full[:, k]))
            # TODO: Ensure this is never 0
            var = np.sum(W * np.power(residuals, 2.0))
            new_std_devs[k] = np.sqrt(var / self.n)
            new_backgrounds[k] = res_wls.params[0]
            new_foregrounds[k] = res_wls.params[1]
        # Update frequencies of each population
        if self.optimize_pop_freqs:
            wght_cnts = np.matmul(W.reshape((self.n * self.z, 1)), self.X_full[:, 1:])
            total = wght_cnts.sum(axis=0)
            freqs = total / np.sum(total)
        else:
            freqs = None
        self.model = JibesModelPy(
            new_backgrounds,
            new_foregrounds,
            new_std_devs,
            freqs,
            blank_prob=self.model.blank_freq,
            n_gems=self.n_gems,
        )

    def _reshape_for_regression(self):
        # Make a new data matrix by replicating each row K times
        to_cat = []
        for i in range(self.n):
            to_cat.append(np.repeat(i, self.z))
        replicated_indices = np.concatenate(to_cat)
        data_new = self.data.counts[replicated_indices, :]
        X_full = np.tile(self.X, (self.n, 1))
        return (data_new, X_full)

    def one_EM_step(self):
        """
        Perform one step of the EM algorithm.
        :returns The current LL of the model.
        """
        if self.posterior is None:
            self._calculate_posterior_by_state()
        print("LL = " + str(self.LL))
        self._maximize_parameters()
        # Calculate this after to ensure we are current with the parameters
        self._calculate_posterior_by_state()
        self.iterations += 1
        return self.LL

    # Note these default parameters are mirrored in rust code and should be updated at the same time
    def perform_EM(self, max_reps=50000, abs_tol=1e-2, rel_tol=1e-7):
        """
        Run the EM algorithm until either max_reps is exceeded or the change in  the LL after
        a run is <= tol or 1 - new/old <= rel_tol
        :param max_reps: Maximum number of repetitions
        :param abs_tol: Absolute difference in LL which would terminate the algorithm
        :param rel_tol: Relative difference in LL which would terminate the algorithm
        :return: The LL of the model after fitting
        """
        last_ll = self.LL
        rep = 0
        while True:
            self.one_EM_step()
            rep += 1
            rel_change = 1.0 - self.LL / last_ll
            abs_change = self.LL - last_ll
            if rep > max_reps:
                print("Did not converge in {} reps".format(max_reps))
                break
            if not np.isneginf(last_ll) and ((abs_change <= abs_tol) or (rel_change <= rel_tol)):
                print("EM algorithm has converged")
                self.converged = True
                break
            last_ll = self.LL
        return self.LL

    def get_snr_dictionary(self):
        """
        Get a dictionary of "snr_TAG_NAME": snr_value
        Returns:
            Dictionary intended for json output
        """
        if not self.converged:
            raise ArithmeticError("Model must converge before SNRs can be reported.")
        snrs = self.model.foreground / self.model.std_devs
        return {"snr_" + ensure_str(x): y for x, y in zip(self.data.column_names, snrs)}
