import random
import pickle
import os
import numpy as np


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.opponent_number = 1 - agent_number  # index for opponent
        self.n_items = params["n_items"]

        # Unpickle the trained model
        # Complications: pickle should work with any machine learning models
        # However, this does not work with custom defined classes, due to the way pickle operates
        # TODO you can replace this with your own model
        path = 'machine_learning_model'
        self.knn_model = pickle.load(open(f'{path}/vectorRegressor.pkl', 'rb'))
        self.model =pickle.load(open(f'{path}/model.pkl', 'rb'))
        self.item0_model =pickle.load(open(f'{path}/item0_model.pkl', 'rb'))
        self.item1_model =pickle.load(open(f'{path}/item1_model.pkl', 'rb'))
        self.price_min = [0.001, 0.001]
        self.price_max = [2.2, 3.5]
        self.max_epoch = 200
        self.item0_embedding = np.array([0.12277592821655194,
                                         0.8848570813366212,
                                         -0.7556286732829098,
                                         0.9490172960627021,
                                         0.6702740700696965,
                                         -1.209413756554651,
                                         -0.2610547783766926,
                                         0.4517198188232259,
                                         -0.8265020776064129,
                                         0.2700059980528833])
        self.item1_embedding = np.array([0.659227815368504,
                                         -0.14133051653919068,
                                         0.08777977176734512,
                                         -1.0989246196354665,
                                         1.2703206502381699,
                                         2.4131725160613238,
                                         -0.7559907972194396,
                                         -0.9461158749689281,
                                         0.3349207822918375,
                                         -0.08573474488666911])
        
        self.last_cavariates = None
        
    def _get_revenue(self, x, price0, price1):
        outcome_proba = self.model.predict_proba(np.concatenate((x, [price0, price1])).reshape(1, -1))
        revenues = (outcome_proba[0][1] * price0 + outcome_proba[0][2] * price1)
        return revenues
        
    def _price(self, x):        
        init_price0 = 1.2836340580063292
        init_price1 = 1.624305842971243

        lr, epsilon = 0.01, 0.0001
        beta1, beta2 = 0.9, 0.999
        m, v = np.array([0, 0]), np.array([0, 0])
        
        best_prices, max_revenue = (0, 0), 0
        curr_prices = np.array([init_price0, init_price1])
        
        for epoch in range(self.max_epoch):
            # Central Difference Gradient Estimater
            f0_x_plus_epsilon = self._get_revenue(x, curr_prices[0] + epsilon, curr_prices[1])
            f0_x_minus_epsilon = self._get_revenue(x, curr_prices[0] - epsilon, curr_prices[1])
            gradient0 = (f0_x_plus_epsilon - f0_x_minus_epsilon) / (epsilon*2)
            
            # Central Difference Gradient Estimater
            f1_x_plus_epsilon = self._get_revenue(x, curr_prices[0], curr_prices[1] + epsilon)
            f1_x_minus_epsilon = self._get_revenue(x, curr_prices[0], curr_prices[1] - epsilon)
            gradient1 = (f1_x_plus_epsilon - f1_x_minus_epsilon) / (epsilon*2)

            # Adam Optimizer
            gradient = np.array([gradient0, gradient1])
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * np.power(gradient, 2)
            m_hat = m / (1 - np.power(beta1, epoch+1))
            v_hat = v / (1 - np.power(beta2, epoch+1))
            curr_prices +=  lr * m_hat / (np.sqrt(v_hat) + 1e-08)

        return curr_prices

    # Given an observation which is #info for new buyer, information for last iteration, and current profit from each time
    # Covariates of the current buyer, and potentially embedding. Embedding may be None
    # Data from last iteration (which item customer purchased, who purchased from, prices for each agent for each item (2x2, where rows are agents and columns are items)))
    # Returns an action: a list of length n_items=2, indicating prices this agent is posting for each item.
    # sale: (item bought, agent bought from, prices)
    def action(self, obs):
        new_buyer_covariates, new_buyer_embedding, last_sale, profit_each_team = obs
        
        if type(new_buyer_embedding) == type(None):
            new_buyer_embedding = self.knn_model.predict([new_buyer_covariates])[0]
        self.last_cavariates = np.concatenate((new_buyer_covariates, \
                                [np.dot(new_buyer_embedding, self.item0_embedding)],\
                                 [np.dot(new_buyer_embedding, self.item1_embedding)]), axis=0)
        
        prices = self._price(self.last_cavariates)
        
        return prices
