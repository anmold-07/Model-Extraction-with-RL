__author__ = 'Fan'

import numpy as np
from sklearn.metrics import accuracy_score

class OfflineMethods:
    RT_in_F = 'retrain in F'
    RT_in_X = 'retrain in X'
    SLV_in_F = 'solve in F'


class OfflineBase(object):
    def __init__(self, oracle, X_ex, y_ex, X_test, y_test, n_features, true_model):
        self.X_ex = X_ex
        self.y_ex = y_ex
        self.X_test = X_test
        self.y_test = y_test

        self.n_features = n_features
        self.oracle = oracle
        self.true_model = true_model
        self.clf2 = None

    def set_clf2(self, clf2):
        assert clf2 is not None
        if hasattr(clf2, 'predict'):
            #print(True)
            self.clf2 = clf2.predict
        else:
            self.clf2 = clf2

    def do(self):
        pass

    def benchmark(self):
        # L_unif
        assert self.clf2 is not None
        X_unif = np.random.uniform(-1, 1, (1000, self.n_features))
        
        #print("X_unif :", X_unif.shape)
        y_unif_ref = self.oracle(X_unif, count=False)
        #print("y_unif_ref :", np.array(list(y_unif_ref)).shape)
        #print("y_unif_ref :", np.array(list(y_unif_ref)))
        y_unif_pred = self.clf2(X_unif)
        #print("y_unif_pred :", y_unif_pred.shape)
        #print("y_unif_pred :", y_unif_pred)

        #print("X_test :", self.X_test.shape)
        y_test_ref = self.oracle(self.X_test, count=False)
        #print("y_test_ref :", np.array(list(y_test_ref)).shape)
        #print("y_test_ref :", np.array(list(y_test_ref)))   
        y_test_pred = self.clf2(self.X_test)
        #print("y_test_pred :", y_test_pred.shape)
        #print("y_test_pred :", np.array(list(y_test_pred)))  

        # print(np.array(list(y_unif_ref)).shape, y_unif_pred.shape)
        # print(np.array(list(y_test_ref)).shape, y_test_pred.shape)

        #L_unif = 1 - accuracy_score(np.array(list(y_unif_ref)).reshape(-1,), y_unif_pred.reshape(-1,))
        #L_test = 1 - accuracy_score(np.array(list(y_test_ref)).reshape(-1,), y_test_pred.reshape(-1,))

        L_unif = 1 - accuracy_score(list(y_unif_ref), list(y_unif_pred))
        L_test = 1 - accuracy_score(list(y_test_ref), list(y_test_pred))

        if -1 in self.y_test:
            if -1 not in y_test_pred:
                y_test_pred = [y if y == 1 else -1 for y in y_test_pred]
            if -1 not in y_test_ref:
                y_test_ref  = [y if y == 1 else -1 for y in y_test_ref]

        print('------')
        print(self.__class__.__name__)
        print('Oracle has a test score of %f' % accuracy_score(self.y_test, y_test_ref))
        print('Extract has a test score of %f' % accuracy_score(self.y_test, y_test_pred))
        #print('The true model parameters are :', type(self.true_model._names))
        print('L_unif_error = %f, L_test_error = %f' % (L_unif, L_test))
        print('------')

        return L_unif, L_test