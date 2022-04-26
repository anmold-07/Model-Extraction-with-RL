#__author__ = 'Anmol'  making necessary modifications to run it on Python-3!
__author__ = 'Fan'

import os
import sys
sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))

import copy
import logging
from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from algorithms.OnlineBase import OnlineBase
from algorithms.libsvmOnline import LibSVMOnline

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from utils.result import Result
from algorithms.OfflineBase import OfflineBase

class ActiveLearning (OfflineBase):
    def __init__(self, ex, retrain_xy, test_xy, n_features, total_budget, n_rounds):
        self.total_budget = total_budget
        self.n_rounds = n_rounds
        self.budget_per_round = int(self.total_budget / self.n_rounds)

        self.ex = ex

        X_ex, y_ex = retrain_xy
        X_test, y_test = test_xy

        # ex.batch_predict = oracle!
        super(self.__class__, self).__init__(
            ex.batch_predict, X_ex, y_ex, X_test, y_test, n_features, ex.get_model
        )

        if 0 in self.y_test:
            self.NEG = 0
        elif -1 in self.y_test:
            self.NEG = -1
        else:
            print('Watch out for test file! Neither 0 nor 1 is included!')

    def do(self):
        # get some initial points until query budgest is satisifed
        # each query results in decreasing the query budget
        # self.ex sei hi tum query badha sakta hai!
        self.ex.collect_up_to_budget(self.budget_per_round * 2)
        x, y = self.ex.pts_near_b, self.ex.pts_near_b_labels

        if len(np.unique(y)) < 2:
            return 1, 1

        # gamma_range = np.logspace(-5, 1, 10, base=10)
        # param_grid = dict(gamma=gamma_range)

        try:
            # cv = StratifiedShuffleSplit(y, n_iter=5, test_size=.2)
            # grid = GridSearchCV(svm.SVC(C=1e5), param_grid=param_grid, cv=cv, n_jobs=-1)
            # grid.fit(x, y)
            # h_best = grid.best_estimator_
            raise ValueError
        except ValueError:
            # get initial model h_best before performing re-training!
            h_best = svm.SVC(C=1e5)
            #h_best = svm.LinearSVC()
            h_best.fit(x, y)

        # perform retraining, and query from the learned model (leading to much less queries)
        for i in range(1, self.n_rounds - 1):   
                                                                                                        # type of random features
            online_ = OnlineBase('',        +1,         self.NEG,       h_best,        self.n_features,            'uniform',          error=.1)
            # find self.budget_per_round number of points near the "learned" hyperplane!
            # Can query as much as I want! No limits this time!
            x_, _ = online_.collect_pts(self.budget_per_round, 50000)  # budget doesn't matter

            # x_ is a list of numpy arrays
            #print(type(x_), x_)
            #print(type(x_[0]), x_[0])

            xx_ = None
            if x_ is None or len(x_) < self.budget_per_round:
                print('Run out of budget when getting x_')
                xx_ = np.random.uniform(-1, 1, (self.budget_per_round - len(x_), self.n_features))

            if x_ is not None and len(x_) > 0:
                x.extend(x_)
                # this is where I am increasing the query count!
                y.extend(self.oracle(x_))

            if xx_ is not None:
                x.extend(xx_)
                y.extend(self.oracle(xx_))

            try:
                # cv = StratifiedShuffleSplit(y, n_iter=5, test_size=.2)
                # grid = GridSearchCV(svm.SVC(C=1e5), param_grid=param_grid, cv=cv, n_jobs=-1)
                # grid.fit(x, y)
                # h_best = grid.best_estimator_
                raise ValueError
            except ValueError:
                h_best = svm.SVC(C=1e5)
                #h_best = svm.LinearSVC()
                h_best.fit(x, y)

            # h_best.fit(x, y)

        # learned hyperplane
        self.set_clf2(h_best)
        return self.benchmark() # (ex.batch_predict, h_.predict, test_x, n_features)


    def do_EAR(self, k_points):
        # get some initial points until query budgest is satisifed
        # each query results in decreasing the query budget
        # self.ex sei hi tum query badha sakta hai!
        self.ex.collect_up_to_budget(self.budget_per_round * 2)
        x, y = self.ex.pts_near_b, self.ex.pts_near_b_labels

        if len(np.unique(y)) < 2:
            return 1, 1

        # get initial model h_best before performing re-training!
        #h_best = svm.SVC(C=1e5)
        h_best = svm.LinearSVC()
        h_best.fit(x, y)

        iter = 0
        while iter + self.budget_per_round * 2 < self.total_budget:
            online_ = OnlineBase('',        +1,         self.NEG,       h_best,         self.n_features,            'uniform',          error=.1)
            # points near the learned hyperplane!
            x_ = online_.collect_pts_for_EAR(k_points, 50000)  # budget doesn't matter
            # print(type(x_))
            # print(type(x_[0]))

            x.extend(x_)
            # this is where I am increasing the count!
            y.extend(self.oracle(x_))

            #h_best = svm.SVC(C=1e5)
            h_best = svm.LinearSVC()
            h_best.fit(x, y)
            iter += 1 

        self.set_clf2(h_best)
        return self.benchmark() # (ex.batch_predict, h_.predict, test_x, n_features)


def run(dataset_name, n_features, n_repeat=1, n_learning_round=5):
    base_dir = os.path.join(os.getcwd(), '../targets/%s/' % dataset_name)
    model_file = os.path.join(base_dir, 'train.scale.model')

    result = Result(dataset_name + '-'+ 'active')
    for repeat in range(0, n_repeat):
        print('Iteration %d of %d'% (repeat, n_repeat - 1))

        # instan/ a classifier with a target model "train.scale.model"
        # keeps track of the number of queries to this model!
        # obj-1, inherits from OnlineBase, LibSVMOnline ka clf1 param not None, OnlineBase ka clf None!
        ex = LibSVMOnline(dataset_name, model_file, (1, -1), n_features, 'uniform', 1e-1)

        # loading test data to evalutae performance of trained/learned model!
        X_test, y_test = load_svmlight_file(os.path.join(base_dir, 'test.scale'), n_features)
        X_test = X_test.todense()

        #print(X_test[0, :].shape)
        #print(ex.batch_predict(X_test))
        #print(ex.clf1(X_test[2, :]))
         
        # iterate through multiple values of alpha=q_by_u in "alpha*(d + 1)"
        for i in result.index:
            q_by_u = result.Q_by_U[i]

            # specifies the budget of the active learner
            # obj-2, inherits from OfflineBase (which has oracle),  OfflineBase ka clf2 None!
            main = ActiveLearning(ex, (None, None), (X_test, y_test), n_features, q_by_u * (n_features + 1), n_learning_round)

            # for Adaptive Retraining!
            #L_unif, L_test = main.do()

            # for Extended Adaptive Retraining!
            # rounds do not matter in this case, as this will keep generating until the query budget "q_by_u * (n_features + 1)" is fulfilled!
            k_points = NUM_ACTIONS
            L_unif, L_test = main.do_EAR(k_points)

            result.L_unif[i].append(L_unif)
            result.L_test[i].append(L_test)
                                      # return self.q
            result.nquery[i].append(  ex.get_n_query() )
              
    print(result)

# number of data points you want to randomly generate in each round of the Adaptive Training process.
NUM_ACTIONS = 10

# choose by uncommenting the dataset you want to run the experiment for!
datasets = {
    #'adult': (123, 'adult'),
    #'australian': (14, 'australian'),
    #'breast-cancer': (10, 'breast-cancer'),
    'circle': (2, 'circle'),
    #'diabetes': (8, 'diabetes'),
    #'fourclass': (2, 'fourclass'),
    #'heart': (13, 'heart'),
    #'moons': (2, 'moons'),
    #'mushrooms': (112, 'mushrooms'),
}

import multiprocessing
if __name__ == '__main__':
    for k, v in datasets.items():
        n_features, dataset_name = v
        p = multiprocessing.Process(target=run, args=(dataset_name, n_features,))
        p.start()