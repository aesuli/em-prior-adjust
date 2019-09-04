import numpy as np


def em(y, posteriors_zero, priors_zero, epsilon=1e-6, positive_class=1):
    """
    Implements the prior correction method based on EM presented in:
    "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure"
    Saerens, Latinne and Decaestecker, 2002
    http://www.isys.ucl.ac.be/staff/marco/Publications/Saerens2002a.pdf

    :param y: true labels of test items, to measure accuracy, precision and recall.
    :param posteriors_zero: posterior probabilities on test items, as returned by a classifier. A 2D-array with shape
    Ø(items, classes).
    :param priors_zero: prior probabilities measured on training set.
    :param epsilon: stopping threshold.
    :param positive_class: class index to be considered the positive one, for precision and recall.
    :return: posteriors_s, priors_s, history: final adjusted posteriors, final adjusted priors, a list of length s
    where each element is a tuple with the step counter, the current priors (as list), the stopping criterium value,
    accuracy, precision and recall.
    """
    s = 0
    priors_s = np.copy(priors_zero)
    posteriors_s = np.copy(posteriors_zero)
    val = 2 * epsilon
    history = list()
    acc = np.mean((y == positive_class) == (posteriors_zero[:, positive_class] > 0.5))
    rec = np.sum(np.logical_and((y == positive_class), (posteriors_zero[:, positive_class] > 0.5))) / np.sum(
        y == positive_class)
    prec = np.sum(np.logical_and((y == positive_class), (posteriors_zero[:, positive_class] > 0.5))) / np.sum(
        posteriors_zero[:, positive_class] > 0.5)
    history.append((s, list(priors_s), 1, acc, prec, rec))
    while not val < epsilon:
        # E step
        ratios = priors_s / priors_zero
        denominators = 0
        for c in range(priors_zero.shape[0]):
            denominators += ratios[c] * posteriors_zero[:, c]
        for c in range(priors_zero.shape[0]):
            posteriors_s[:, c] = ratios[c] * posteriors_zero[:, c] / denominators

        acc = np.mean((y == positive_class) == (posteriors_s[:, positive_class] > 0.5))
        rec = np.sum(np.logical_and((y == positive_class), (posteriors_s[:, positive_class] > 0.5))) / np.sum(
            y == positive_class)
        prec = np.sum(np.logical_and((y == positive_class), (posteriors_s[:, positive_class] > 0.5))) / np.sum(
            posteriors_s[:, positive_class] > 0.5)
        priors_s_minus_one = priors_s.copy()

        # M step
        priors_s = posteriors_s.mean(0)

        # check for stop
        val = 0
        for i in range(len(priors_s_minus_one)):
            val += abs(priors_s_minus_one[i] - priors_s[i])
        s += 1
        history.append((s, list(priors_s), val, acc, prec, rec))

    return posteriors_s, priors_s, history
