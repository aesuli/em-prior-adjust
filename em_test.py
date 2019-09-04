import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC

from em import em


def em_experiment(ax, clf, X_tr, y_tr, X_te, y_te, y_min=0, y_max=1.0):
    mlb = MultiLabelBinarizer()
    mlb.fit(np.expand_dims(np.hstack((y_tr, y_te)), 1))
    y_tr_bin = mlb.transform(np.expand_dims(y_tr, 1))
    y_te_bin = mlb.transform(np.expand_dims(y_te, 1))
    train_priors = np.mean(y_tr_bin, 0)
    test_priors = np.mean(y_te_bin, 0)

    print("Fitting", clf)

    clf.fit(X_tr, y_tr)
    test_posteriors = clf.predict_proba(X_te)
    posteriors_test_priors = np.mean(test_posteriors, axis=0)

    print('train priors', train_priors, sep='\n')
    print('test priors', test_priors, sep='\n')
    print('posteriors mean', posteriors_test_priors, sep='\n')
    print()

    em_test_posteriors, em_test_priors, history = em(y_te, test_posteriors, train_priors)

    em_prior = [p[1] for _, p, _, _, _, _ in history]
    accuracy = [a for _, _, _, a, _, _ in history]
    f1 = [2 * p * r / (p + r) if p + r > 0 else 0 for _, _, _, _, p, r in history]
    ax.set_ylim([y_min, y_max])
    ax.plot(range(len(accuracy)), accuracy, linestyle='-.', color='m', label='accuracy')
    ax.plot(range(len(f1)), f1, linestyle='--', color='#dd9f00', label='f1')
    ax.plot(range(len(em_prior)), em_prior, color='b', label='em pr')
    ax.hlines([train_priors[1]], 0, len(em_prior) - 1, colors=['r'], linestyles=[':'], label='train pr')
    ax.hlines([posteriors_test_priors[1]], 0, len(em_prior) - 1, colors=['#b5651d'], linestyles=['-.'], label='clf pr')
    ax.hlines([test_priors[1]], 0, len(em_prior) - 1, colors=['g'], linestyles=['--'], label='test pr')

    ax.set()
    ax.grid()

    print('Results')
    print('prior from:   train test  post  em')
    for i, (a, b, c, d) in enumerate(
            zip(train_priors, test_priors, posteriors_test_priors, em_test_priors)):
        print(f'{i:11d} - {a:3.3f} {b:3.3f} {c:3.3f} {d:3.3f}')

    return posteriors_test_priors[1], em_test_priors[1], accuracy[0], accuracy[-1], f1[0], f1[-1]


def batch(batch_name, tr_prevalences, te_prevalences, y_min=0, y_max=1.0):
    for name, clf in [('Multinomial Bayes', MultinomialNB()),
                      ('Calibrated Multinomial Bayes', CalibratedClassifierCV(MultinomialNB())),
                      ('Linear SVM', SVC(probability=True, kernel='linear')),
                      ('Calibrated Linear SVM', CalibratedClassifierCV(SVC(probability=True, kernel='linear'))),
                      ('Logistic Regression', LogisticRegression()),
                      ('Calibrated Logistic Regression', CalibratedClassifierCV(LogisticRegression())),
                      ('Random Forest', RandomForestClassifier()),
                      ('Calibrated Random Forest', CalibratedClassifierCV(RandomForestClassifier()))]:
        fig, axs = plt.subplots(len(tr_prevalences), len(te_prevalences))
        fig.suptitle(name)
        fig.set_size_inches(20, 20)
        axs_iter = iter(axs.flat)
        for tr_pr in tr_prevalences:
            for te_pr in te_prevalences:
                ax = next(axs_iter)
                pos_count = y.sum()
                neg_count = len(y) - pos_count
                tr_pos = int(pos_count * tr_pr / (tr_pr + te_pr))
                te_pos = pos_count - tr_pos
                tr_neg = int(tr_pos / tr_pr * (1 - tr_pr))
                te_neg = int(te_pos / te_pr * (1 - te_pr))
                if tr_neg + te_neg > neg_count:
                    factor = neg_count / (tr_neg + te_neg)
                    tr_neg = int(tr_neg * factor)
                    te_neg = neg_count - tr_neg
                    tr_pos = int(tr_pos * factor)
                    te_pos = int(te_pos * factor)
                tr_idx = list()
                for i in range(len(y)):
                    if y[i] == 1:
                        tr_idx.append(i)
                        if len(tr_idx) == tr_pos:
                            break
                for i in range(len(y)):
                    if y[i] == 0:
                        tr_idx.append(i)
                        if len(tr_idx) == tr_pos + tr_neg:
                            break
                te_idx = list()
                for i in range(len(y)):
                    if y[i] == 1 and i not in tr_idx:
                        te_idx.append(i)
                        if len(te_idx) == te_pos:
                            break
                for i in range(len(y)):
                    if y[i] == 0 and i not in tr_idx:
                        te_idx.append(i)
                        if len(te_idx) == te_pos + te_neg:
                            break
                X_tr = X[tr_idx]
                y_tr = y[tr_idx]
                X_te = X[te_idx]
                y_te = y[te_idx]

                posterior_prev, em_prev, posterior_acc, em_acc, posterior_f1, em_f1 = em_experiment(ax, clf, X_tr, y_tr,
                                                                                                    X_te, y_te, y_min,
                                                                                                    y_max)
                ax.set_title(
                    f'Test pr = {te_pr:3.3f}, Train pr = {tr_pr:3.3f}\n' +
                    f'Post pr = {posterior_prev:3.3f},    EM pr = {em_prev:3.3f}\n' +
                    f'Post acc= {posterior_acc:3.3f},    EM acc= {em_acc:3.3f}\n' +
                    f'Post f1= {posterior_f1:3.3f},    EM f1= {em_f1:3.3f}')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', ncol=3)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs('plots', exist_ok=True)
        fig.savefig(os.path.join('plots', batch_name + '_' + name + '.pdf'))


if __name__ == '__main__':
    dataset = fetch_20newsgroups_vectorized()
    X = dataset.data
    y = dataset.target

    print('transforming 20NG into a binary classification problem in which any post under the comp.* tree is positive')
    pos_idx = list()
    for i, name in enumerate(dataset.target_names):
        if name.startswith('comp'):
            pos_idx.append(i)
    y = np.isin(y, pos_idx) * 1

    print(f'{sum(y)} positive examples across {len(y)} total examples')

    tr_prevalences = np.arange(0.1, 1, 0.2)
    te_prevalences = np.arange(0.1, 1, 0.2)
    batch('em', tr_prevalences, te_prevalences)

    tr_prevalences = [0.001, 0.002, 0.005, 0.01, 0.02]
    te_prevalences = [0.001, 0.002, 0.005, 0.01, 0.02]
    batch('low_pr_em', tr_prevalences, te_prevalences, 0, 0.025)

    tr_prevalences = np.arange(0.1, 1, 0.2)
    te_prevalences = [0.001, 0.002, 0.005, 0.01, 0.02]
    batch('al_low_pr_em', tr_prevalences, te_prevalences, 0, 0.025)
