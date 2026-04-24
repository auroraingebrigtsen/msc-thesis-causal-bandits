# This file contains code adapted from:
#
# Reda Alami (2018)
# Restart Bayesian Online Change Point Detection
# https://github.com/Ralami1859/BayesianOnlineChange-pointDetection-python-codes-
#


from .base import BaseCPD
import numpy as np

class RBOCPD(BaseCPD):
    def __init__(self, gamma: float):
        self.gamma = gamma
        self.alphas = np.array([1])
        self.betas = np.array([1])
        self.forecaster_distribution = np.array([1])
        self.pseudo_dist = np.array([1])
        self.like1 = 1
        self.drift_detected = False
        self.restarted = False

    def update(self, reward: float) -> None:
        if self.restarted:
            self._update_forecaster_distribution_m(reward=reward)
        else:
            self._update_laplace_prediction(reward=reward)
        estimated_best_expert = np.argmax(self.forecaster_distribution)
        self.drift_detected = not(estimated_best_expert == 0)


    def _update_forecaster_distribution(self, reward):
        """Updating the forecaster distribution using the message passing algorithm"""
        if reward == 1:
            likelihood = np.divide(self.alphas, self.alphas + self.betas)
        else:
            likelihood = np.divide(self.betas, self.alphas + self.betas)
        forecaster_distribution0 = self.gamma*np.dot(likelihood, np.transpose(self.forecaster_distribution)) # Creating new Forecaster 
        forecaster_distribution = (1-self.gamma)*likelihood*self.forecaster_distribution # update the previous forecasters 
        forecaster_distribution = np.append(forecaster_distribution,forecaster_distribution0) # Including the new forecaseter into the previons ones
        self.forecaster_distribution = forecaster_distribution/np.sum(forecaster_distribution) # Normalization for numerical purposes

    def _update_forecaster_distribution_m(self, reward):
        """Updating the forecaster distribution using the message passing algorithm with a modified prior (q)"""
        if reward == 1:
            likelihood = np.divide(self.alphas, self.alphas + self.betas)
        else:
            likelihood = np.divide(self.betas, self.alphas + self.betas)
        pseudo_w0 = self.gamma*self.like1*np.sum(self.pseudo_dist) # Using the simple prior
        pseudo_dist = self.like1*self.pseudo_dist
        forecaster_distribution0 = pseudo_w0 # Creating new Forecaster
        forecaster_distribution = (1-self.gamma)*likelihood*self.forecaster_distribution # update the previous forecasters
        forecaster_distribution = np.append(forecaster_distribution,forecaster_distribution0) # Including the new forecaseter into the previons ones
        self.forecaster_distribution = forecaster_distribution/np.sum(forecaster_distribution) # Normalization for numerical purposes

        pseudo_dist = np.append(pseudo_dist,pseudo_w0)
        self.pseudo_dist = pseudo_dist/np.sum(pseudo_dist) # Normalization for numerical purposes

    def _update_laplace_prediction(self, reward):
        self.alphas[:] += reward
        self.betas[:] += (1-reward)
        self.alphas = np.append(self.alphas,1) # Creating new Forecaster
        self.betas = np.append(self.betas,1) # Creating new Forecaster

    
    def reset(self) -> None:
        self.alphas = np.array([1])
        self.betas = np.array([1])
        self.forecaster_distribution = np.array([1])
        self.pseudo_dist = np.array([1])
        self.like1 = 1
        self.drift_detected = False