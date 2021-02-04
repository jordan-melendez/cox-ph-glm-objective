import numpy as np
from numpy import ndarray

import numba as nb
from numba import jit, prange
from typing import Optional, List, Iterable, Tuple, Union

ArrayLike = Union[ndarray, List, Iterable]


@jit(nopython=True)
def argsort_by_magnitude_with_negs_last(a: ArrayLike) -> ndarray:
    r"""Compute argsort by magnitude, with equal-magnitude negative values placed after positive ones.

    Parameters
    ----------
    a : array, shape = (N,)
        An array of integers or floats to be sorted.

    Returns
    -------
    sorter : array, shape = (N,)
        An array that will sort `a` by magnitude, with equal-magnitude negative values placed after positive ones.

    Examples
    --------
    >>> a = np.array([20, 0, -4, 4, -2, 1, 2, -4, -2, 20])
    >>> a[argsort_by_magnitude_with_negs_last(a)]
    array([ 0,  1,  2, -2, -2,  4, -4, -4, 20, 20])
    """
    # Jit and pure numpy appear to behave differently for argsort.
    # `mergesort` appears to make the result work okay, but this should be investigated further.
    kind = "mergesort"
    argsort_1 = np.argsort(-a, kind=kind)  # Descending
    argsort_2 = np.argsort(np.abs(a[argsort_1]), kind=kind)
    return argsort_1[argsort_2]


@jit(
    nopython=True,
    locals=dict(out=nb.float64, weights_sum=nb.float64, log_sum_exp_offset=nb.float64),
)
def partial_log_likelihood(
        t: ArrayLike,
        pred: ArrayLike,
        weights: Optional[ArrayLike] = None,
        sorter: Optional[ArrayLike] = None,
) -> float:
    r"""The partial log likelihood as defined in the Cox PH model

    Parameters
    ----------
    t : array
        The times of the events, sorted from smallest to largest by magnitude.
        If the event has not occurred, t should be negative and indicates the censoring time.
    pred : array
        The predictions of the model. In the standard Cox linear regression, this is x.T @ beta.
    weights : array or None
    sorter : list, optional
        A list of indices that would sort t by magnitude. If None, then this assumes t is already sorted (and that
        pred has been arranged to match)

    Returns
    -------
    float
        The partial log likelihood
    """
    assert len(t) == len(pred)

    exp_p_sum = 0
    n_data = len(t)
    if sorter is None:
        sorter = np.arange(n_data)
    sort_backwards = sorter[::-1]

    log_sum_exp_offset = 0.0
    for ii in range(n_data):
        this_pred = pred[sorter[n_data - ii - 1]]
        if this_pred > 0:
            log_sum_exp_offset = this_pred
            break

    prev_t = np.inf
    out = 0.0
    weights_sum = 0.0
    # Go backwards for numerical stability. This allows exp_p_sum to be incremented,
    # rather than summed first and pieces individually subtracted off. When adding up front, and
    # then subtracting piece by piece, this can make exp_p_sum be zero or negative when it should not be.
    for n, idx in enumerate(sort_backwards):
        t_i = t[idx]
        p_i = pred[idx]
        w_i = 1.0 if weights is None else weights[idx]

        if abs(t_i) < abs(prev_t):
            # The exponential sum would be weighted also. See the likelihood proposed in
            # https://www.jstor.org/stable/2531402?seq=1 (Cain & Lange, 1984)
            # Also described in the h2o docs on Cox PH:
            # http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/coxph.html
            # But as of Oct 2020 there is a typo in the first partial likelihood equation
            # The denominator should be exponentiated by the weight. The subsequent Eqs are correct.

            exp_p_sum += w_i * np.exp(p_i - log_sum_exp_offset)
            # Also include all tied values:
            jj = 1
            while n + jj < n_data and abs(t[sort_backwards[n + jj]]) == abs(t_i):
                idx_new = sort_backwards[n + jj]
                w_new = 1.0 if weights is None else weights[idx_new]
                exp_p_sum += w_new * np.exp(pred[idx_new] - log_sum_exp_offset)
                jj += 1

        if t_i > 0:
            # Computing log likelihood, not its negative
            out += w_i * (p_i - log_sum_exp_offset - np.log(exp_p_sum))
            weights_sum += w_i

        prev_t = t_i
    return out


@jit(
    nopython=True,
    locals=dict(r_k=nb.float64, s_k=nb.float64, log_sum_exp_offset=nb.float64),
)
def grad_partial_log_likelihood(
        t: ArrayLike,
        pred: ArrayLike,
        weights: Optional[ArrayLike] = None,
        sorter: Optional[ArrayLike] = None,
) -> Tuple[ndarray, ndarray]:
    r"""The partial log likelihood as defined in the Cox PH model

    The proportional hazard model assumes h(t|x) = h_0(t) * exp(f(x)). The predictions are assumed to be f(x).

    Parameters
    ----------
    t : array
        The times of the events, sorted from smallest to largest by magnitude.
        If the event has not occurred, t should be negative and indicates the censoring time.
    pred : array
        The predictions of the model f(x). In the standard Cox linear regression, this is x.T @ beta.
    weights : array or None
        The weights of each datum
    sorter : list, optional
        A list of indices that would sort t by magnitude. If None, then this assumes t is already sorted (and that
        pred has been arranged to match)

    Returns
    -------
    grad : array
        The gradient of the Cox partial log likelihood
    hess : array
        The Hessian of the Cox partial log likelihood
    """
    n_data = len(t)
    if sorter is None:
        sorter = np.arange(n_data)
    sorter_reverse = sorter[::-1]

    log_sum_exp_offset = 0.0
    for ii in range(n_data):
        this_pred = pred[sorter[n_data - ii - 1]]
        if this_pred > 0:
            log_sum_exp_offset = this_pred
            break

    # Compute the cumulative sum of the exp(preds) in reverse order, so it's like we are subtracting off
    # the exp(pred_i) as the index increments. So make sure that the first
    # value of the vector is the full sum, and the pred values corresponding to small t are the ones
    # removed first, as the index increases. If there are any ties in magnitude, don't "subtract" off
    # any of the pred values until the tie is broken
    exp_p_cum_sum = np.zeros(n_data, dtype=np.float64)
    prev_abs_t_ii = np.inf
    for n, idx in enumerate(sorter_reverse):
        abs_t_ii = abs(t[idx])
        w_ii = 1.0 if weights is None else weights[idx]
        if abs_t_ii < prev_abs_t_ii:
            exp_p_sum_ii = w_ii * np.exp(pred[idx] - log_sum_exp_offset)
            # Include all values that are tied as well (if they exist)
            jj = 1
            while (n + jj < n_data) and abs_t_ii == abs(t[sorter_reverse[n + jj]]):
                idx_new = sorter_reverse[n + jj]
                w_new = 1.0 if weights is None else weights[idx_new]
                exp_p_sum_ii += w_new * np.exp(pred[idx_new] - log_sum_exp_offset)
                jj += 1

            exp_p_cum_sum[n] = exp_p_sum_ii + exp_p_cum_sum[n - 1]
        elif abs_t_ii == prev_abs_t_ii:
            # Ties should use the previously computed value
            exp_p_cum_sum[n] = exp_p_cum_sum[n - 1]
        else:
            # nopython mode is picky about string formatting. Just print some info.
            # Maybe this can be logged in the future.
            print("Count:", n)
            print("Index:", idx)
            print("Previous magnitude:", prev_abs_t_ii)
            print("Current magnitude:", abs_t_ii)
            raise ValueError("t is not sorted.")

        prev_abs_t_ii = abs_t_ii
    exp_p_cum_sum = exp_p_cum_sum[::-1]

    # Set up variables and start computing gradients
    r_k = 0.0
    s_k = 0.0
    prev_abs_t_i = -1
    grad = np.zeros(n_data, dtype=np.float64)
    hess = np.zeros(n_data, dtype=np.float64)
    for n, idx in enumerate(sorter):
        p_i = pred[idx]
        # Avoid creating a new weights array if not needed
        w_i = 1.0 if weights is None else weights[idx]
        # TODO: Think about if weight should be attached here, or below
        exp_p_i_offset = w_i * np.exp(p_i - log_sum_exp_offset)
        t_i = t[idx]
        abs_t_i = np.abs(t_i)
        exp_p_sum = exp_p_cum_sum[n]  # Already set up all sums above

        if t_i > 0 and abs_t_i > prev_abs_t_i:
            n_ties = 1 * w_i
            while n + n_ties < n_data and t[sorter[n + n_ties]] == t_i:
                w_tied_i = 1.0 if weights is None else weights[sorter[n + n_ties]]
                n_ties += 1 * w_tied_i
            r_k += n_ties / exp_p_sum
            s_k += n_ties / exp_p_sum ** 2

        # Removed offset from numerator and denominator, so it should be more stable
        # TODO: See above todo, may be over-counting weights
        grad[idx] = w_i * ((t_i > 0) - exp_p_i_offset * r_k)
        hess[idx] = w_i * exp_p_i_offset * (exp_p_i_offset * s_k - r_k)
        prev_abs_t_i = abs_t_i
    return grad, hess


class CoxMixin:
    def __init__(self, auto_sort=True):
        self.auto_sort = auto_sort
        self._arg_sorter = None

    def argsort_by_magnitude(self, target: ArrayLike) -> ndarray:
        """Returns the indices that would sort target by magnitude (sign ignored).

        Parameters
        ----------
        target :
            The target times, with negative values indicating censoring times.

        Returns
        -------
        indices :
            The indices that would sort target by magnitude
        """
        target = np.asarray(target)
        if self.auto_sort:
            # target = np.asarray(target)
            #         if (
            #             True
            #             or not hasattr(self, "_argsort_target")
            #             or len(self._argsort_target) != len(target)
            #         ):
            #             self._argsort_target = argsort_by_magnitude_with_negs_last(target)
            #         return self._argsort_target

            # Setting an attribute *might* be causing issues with parallelization
            # i.e., it could get overwritten in between the assign and return steps above.
            # so instead just return an array
            return argsort_by_magnitude_with_negs_last(target)
        else:
            return np.arange(target.shape[0], dtype=int)


class CoxPHObjective(CoxMixin):
    def calc_ders_range(self, approxes, targets, weights):
        # approxes, targets, weights are indexed containers of floats
        # (containers which have only __len__ and __getitem__ defined).
        # weights parameter can be None.
        #
        # To understand what these parameters mean, assume that there is
        # a subset of your dataset that is currently being processed.
        # approxes contains current predictions for this subset,
        # targets contains target values you provided with the dataset.
        #
        # This function should return a list of pairs (der1, der2), where
        # der1 is the first derivative of the loss function with respect
        # to the predicted value, and der2 is the second derivative.
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        approxes = np.asarray(approxes, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
        sorter = self.argsort_by_magnitude(targets)

        # Regardless of the choice of is_max_optimal in the metric, it looks like this should act like
        # a *likelihood*, not its negative. So don't add a negative sign here.
        grad, hess = grad_partial_log_likelihood(
            t=targets, pred=approxes, weights=weights, sorter=sorter
        )
        return list(zip(grad, hess))


class CoxPHMetric(CoxMixin):
    def is_max_optimal(self):
        # Returns whether great values of metric are better
        return False

    def evaluate(self, approxes, target, weight):
        # approxes is a list of indexed containers
        # (containers with only __len__ and __getitem__ defined),
        # one container per approx dimension.
        # Each container contains floats.
        # weight is a one dimensional indexed container.
        # target is a one dimensional indexed container.

        # weight parameter can be None.
        # Returns pair (error, weights sum)
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        approx = np.asarray(approx, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if weight is not None:
            weight = np.asarray(weight, dtype=np.float64)

        # Anytime there are categorical features, target is randomly sorted.
        # This caches the sorter so it doesn't have to be recomputed every time
        sorter = self.argsort_by_magnitude(target)

        # Make the log likelihood negative:
        error_sum = -partial_log_likelihood(
            t=target, pred=approx, weights=weight, sorter=sorter
        )
        if weight is None:
            weight_sum = len(approx)
        else:
            weight_sum = np.sum(weight)
        return error_sum, weight_sum

    def get_final_error(self, error, weight):
        # Returns final value of metric based on error and weight
        return error / (weight + 1e-38)
