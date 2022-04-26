# Model Extraction Attacks via Active Reinforcement Learning (RL) Policies:
__author__ = 'Anmol'

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

# importing deep learning stufF!
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

model = None
optimizer = None
# start with the architecture!
class DQN(nn.Module):
    """A simple deep Q-network implementation that computes Q-values for each action given an input state vector.
    """
    def __init__(self, state_dim, action_dim, hidden_size=32):
        super(DQN, self).__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.state2action = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        state = torch.tanh(self.state_encoder(x))
        return self.state2action(state)

# Choose actions for a given state=state_vector!
def epsilon_greedy(state_vector, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy
    Args:   state_vector (torch.FloatTensor): extracted vector representation
            epsilon (float): the probability of choosing a random command
    Returns:
            (int): indices for the action to take
    """
    randNumber, maxQ = np.random.random_sample((1)).item(), float('-Inf')
    if randNumber<epsilon:
        action_index = np.random.randint(0, NUM_ACTIONS)
    else:
        q_value_cur_state = model(state_vector)  
        action_index = torch.argmax(q_value_cur_state[0])
                    
    return int(action_index)

def deep_q_learning(torch_current_state_vector, action_index, reward, torch_next_state_vector, terminal):
    """Updates the weights of the DQN for a given transition
    Args:
        torch_current_state_vector (torch.FloatTensor): vector representation of current state
        action_index (int): index of the current action
        reward (float): the immediate reward the agent recieves from playing current command
        torch_next_state_vector (torch.FloatTensor): vector representation of next state
        terminal (bool): True if this epsiode is over
    Returns:
        None
    """
    with torch.no_grad():
        q_values_action_next = model(torch_next_state_vector)
    maxq_next = q_values_action_next.max()

    # Current Q-value 
    q_value_cur_state = model(torch_current_state_vector)
    q_val_cur = q_value_cur_state[action_index]

    maxQ = 0.0 if terminal else maxq_next
    y = reward + GAMMA*maxQ # Target

    loss = 1/2 * (y - q_val_cur)**2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
    
    def DQN_env_step(self, action_index, h_best, x, y, counter):
        # error for current model
        self.set_clf2(h_best)

        # select the kind of rewards you want to give to agent (either rewards on test datat or random uniform data)
        """
        X_unif = np.random.uniform(-1, 1, (1000, self.n_features))
        old_y_unif_pred = self.clf2(X_unif)
        old_y_unif_ref = self.oracle(X_unif, count=False)
        old_L_test = 1 - accuracy_score(list(old_y_unif_ref), list(old_y_unif_pred)) # error
        """
        old_y_test_ref = self.oracle(self.X_test, count=False)
        old_y_test_pred = self.clf2(self.X_test)
        old_L_test = 1 - accuracy_score(list(old_y_test_ref), list(old_y_test_pred)) # error
    
        online_ = OnlineBase('',        +1,         self.NEG,       h_best,         self.n_features,            'uniform',          error=.1)

        x_ = online_.collect_pts_for_DQN(NUM_ACTIONS, action_index, 50000)  
        x.extend(x_)
        # this is where I am increasing the count!
        y.extend(self.oracle(x_))

        # h_best = svm.SVC(C = 1e5)
        new_h_best = svm.LinearSVC()
        new_h_best.fit(x, y)

        numpy_next_state = np.concatenate((new_h_best.coef_[0], new_h_best.intercept_))
        #print(numpy_next_state)
 
        # error for updated model
        self.set_clf2(new_h_best)

        # select the kind of rewards you want to give to agent (either rewards on test datat or random uniform data)
        """
        new_y_unif_pred = self.clf2(X_unif)
        new_y_unif_ref = self.oracle(X_unif, count=False)
        new_L_test = 1 - accuracy_score(list(new_y_unif_ref), list(new_y_unif_pred)) # error
        """
        new_y_test_ref = self.oracle(self.X_test, count = False)
        new_y_test_pred = self.clf2(self.X_test)
        new_L_test = 1 - accuracy_score(list(new_y_test_ref), list(new_y_test_pred))
    
        # reward scaling doesn't really matter!
        SCALE_REWARDS = 10**5
        reward = (old_L_test - new_L_test)*SCALE_REWARDS

        if counter < self.total_budget:
            terminal = False
        else:    
            terminal = True

        return numpy_next_state, reward, terminal, new_h_best


    def run_epsiodes(self, for_training):   
        """
        x, y : Lists
        current_room_desc : np.array
        next_action_index : int
        """ 
        epsilon = TRAINING_EP if for_training else TESTING_EP
        epi_reward = None
        counter = 0

        self.ex.collect_up_to_budget(self.budget_per_round * 2)
        x, y = self.ex.pts_near_b, self.ex.pts_near_b_labels

        if len(np.unique(y)) < 2:
            return 1, 1

        # get initial model h_best before performing re-training!
        # h_best = svm.SVC(C=1e5)
        h_best = svm.LinearSVC()
        h_best.fit(x, y)
        
        numpy_current_state = np.concatenate((h_best.coef_[0], h_best.intercept_))
        #print(numpy_current_state, False, h_best, x, y)
        terminal = False
        current_model = h_best
        counter = self.budget_per_round * 2

        """ 
        # although numpy_current_state is uniquely related to current_model, keeping both for convenience!
        numpy_current_state, terminal, current_model, x, y = self.DQN_initialize()
        counter = self.budget_per_round * 2
        """

        while not terminal:
            # increase current budget count!
            counter += 1

            # Choose next action and execute
            # concatenated input parameters!
            current_state = numpy_current_state
            torch_current_state_vector = torch.FloatTensor(current_state)

            # take action
            action_index = epsilon_greedy(torch_current_state_vector, epsilon)

            # execute action
            numpy_next_state, reward, terminal, new_model= self.DQN_env_step(action_index, copy.deepcopy(current_model), x, y, counter) 

            # concatenated input parameters!
            next_state = numpy_next_state 
            torch_next_state_vector = torch.FloatTensor(next_state)       

            if for_training:
                # update Q-function
                deep_q_learning(torch_current_state_vector, action_index, reward, torch_next_state_vector, terminal) 

            if not for_training:
                if epi_reward==None:
                    epi_reward = (GAMMA**(counter-self.budget_per_round * 2))*reward
                else:
                    epi_reward += (GAMMA**(counter-self.budget_per_round * 2))*reward

            current_model = copy.deepcopy(new_model)
            numpy_current_state = copy.deepcopy(numpy_next_state)

        if not for_training:
            return (current_model, epi_reward)

        return current_model

    def run_train(self):
        for iter in range(NUM_EPISODE_TRAIN):
            print("training iteration: ", iter)
            current_model = self.run_epsiodes(for_training = True)
            #self.set_clf2(current_model)
            #print(self.benchmark())

        return current_model

    def run_DQN_training(self):
        global model
        global optimizer
        model = DQN(self.n_features+1, NUM_ACTIONS)
        optimizer = optim.SGD(model.parameters(), lr=ALPHA)

        current_model = self.run_train()
        self.set_clf2(current_model)
        print("---- Results for Trained Model: ----")
        return self.benchmark()

    def run_DQN_inference(self):
        rewards = []
        for _ in range(NUM_EPISODE_TEST):
            learned_model, obtained_reward = self.run_epsiodes(for_training = False)
            rewards.append(obtained_reward)

        print(rewards)
        print("---- Results for Inference Model: ----")
        self.set_clf2(learned_model)
        return self.benchmark()


NUM_EPISODE_TRAIN = 20
NUM_EPISODE_TEST = 1
NUM_ACTIONS = 10
TRAINING_EP = 0.5   # epsilon-greedy parameter for training
TESTING_EP = 0.05   # epsilon-greedy parameter for testing
GAMMA = 0.99        # discounted factor
ALPHA = 0.01        # learning rate

def train_DQN(dataset_name, n_features, n_repeat=1, n_learning_round=5):
    base_dir = os.path.join(os.getcwd(), '../targets/%s/' % dataset_name)
    model_file = os.path.join(base_dir, 'train.scale.model')

    result_train = Result(dataset_name + '-'+ 'active for learning')
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
        for i in result_train.index:
            q_by_u = result_train.Q_by_U[i]

            # specifies the budget of the active learner
            # obj-2, inherits from OfflineBase (which has oracle),  OfflineBase ka clf2 None!
            main = ActiveLearning(ex, (None, None), (X_test, y_test), n_features, q_by_u * (n_features + 1), n_learning_round)
            L_unif, L_test = main.run_DQN_training()
            result_train.L_unif[i].append(L_unif)
            result_train.L_test[i].append(L_test)
                                    # return self.q
            result_train.nquery[i].append(  ex.get_n_query() )

    print(result_train)

    # reaplace the dataset name on which you want to perform inference on. Ideally, not the one you trained on that also has the same number of features as the one trained on!
    dataset_name = 'circle'
    result_inference = Result(dataset_name + '-'+ 'active for inference')
    base_dir = os.path.join(os.getcwd(), '../targets/%s/' % dataset_name)
    model_file = os.path.join(base_dir, 'train.scale.model')
    ex = LibSVMOnline(dataset_name, model_file, (1, -1), n_features, 'uniform', 1e-1)
    X_test, y_test = load_svmlight_file(os.path.join(base_dir, 'test.scale'), n_features)
    X_test = X_test.todense()
    for i in result_inference.index:
        q_by_u = result_inference.Q_by_U[i]

        # specifies the budget of the active learner
        # obj-2, inherits from OfflineBase (which has oracle),  OfflineBase ka clf2 None!
        main = ActiveLearning(ex, (None, None), (X_test, y_test), n_features, q_by_u * (n_features + 1), n_learning_round)
        L_unif, L_test = main.run_DQN_inference()
        result_inference.L_unif[i].append(L_unif)
        result_inference.L_test[i].append(L_test)

                                # return self.q
        result_inference.nquery[i].append(  ex.get_n_query() )

    print(result_inference)

# choose by uncommenting the dataset you want to run the experiment for!
datasets = {
    #'adult': (123, 'adult'),
    #'australian': (14, 'australian'),
    #'breast-cancer': (10, 'breast-cancer'),
    #'circle': (2, 'circle'),
    #'diabetes': (8, 'diabetes'),
    #'fourclass': (2, 'fourclass'),
    #'heart': (13, 'heart'),
    'moons': (2, 'moons'),
    #'mushrooms': (112, 'mushrooms'),
}

import multiprocessing
if __name__ == '__main__':
    for k, v in datasets.items():
        n_features, dataset_name = v
        p = multiprocessing.Process(target=train_DQN, args=(dataset_name, n_features,))
        p.start()