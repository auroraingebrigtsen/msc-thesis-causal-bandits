# This file contains code adapted from:
#
# Reda Alami (2018)
# Restart Bayesian Online Change Point Detection
# https://github.com/Ralami1859/BayesianOnlineChange-pointDetection-python-codes-
#


from .base import BaseCPD
import numpy as np

class RBOCPD(BaseCPD):
    def __init__(self, horizion: int):
        self.horizon = horizion
        self.gamma = 1/horizion  # Switching Rate
        self.alphas = np.array([1])
        self.betas = np.array([1])
        self.forecaster_distribution = np.array([1])
        self.change_point_estimation = np.array([])
        self.restart = 1  # Position of last restart

    def update(self, reward: float) -> None:
        estimated_best_expert = np.argmax(self.forecaster_distribution)
        if not (estimated_best_expert == 0):
            self.alphas = np.array([1])
            self.betas = np.array([])
            self.forecaster_distribution = np.array([1])
            self.restart += 1
        
        self.change_point_estimation = np.append(self.change_point_estimation, self.restart+1)
        self._update_forecaster_distribution(reward=reward)
        self._updateLaplacePrediction(reward=reward)

    
    def _update_forecaster_distribution(self, reward):
        if reward == 1:
            likelihood = np.divide(self.alphas,self. alphas + self.betas)
        else:
            likelihood = np.divide(self.betas, self.alphas + self.betas)
        
        forecaster_distribution_0 = self.gamma*np.dot(likelihood, np.transpose(self.forecaster_distribution))  # Creating new Forecaster 
        forecaster_distribution = (1-self.gamma)*likelihood*self.forecaster_distribution # update the previous forecasters 
        forecaster_distribution = np.append(forecaster_distribution, forecaster_distribution_0) # Including the new forecaseter into the previons ones
        forecaster_distribution = forecaster_distribution/np.sum(forecaster_distribution) # Normalization for numerical purposes
        self.forecaster_distribution = forecaster_distribution

    def _updateLaplacePrediction(self, reward):
        self.alphas[:] += reward
        self.betas[:] += 1-reward
        self.alphas = np.append(self.alphas,1) # Creating new Forecaster
        self.betas = np.append(self.betas,1) # Creating new Forecaster

    def is_change_point(self) -> bool:
        # DO some logic
        cp = True
        if cp:
            print("Change Point Detected")
        return cp
    
    def reset(self) -> None:
        pass