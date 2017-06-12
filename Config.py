import numpy as np

class Config:
    """Basic DQN Training Configuration"""
    def __init__(self):

        # Environment observation dimentions
        self.ObservationDimentions = [84,84,1]
        self.ActionsCount = 4
        self.TerminateState =  np.zeros(self.ObservationDimentions)

        # Size of the minibatch for training?
        self.MiniBatchSize = 32

        # Optimiser learning rate.
        self.LearningRate = 0.003

        self.LossFuctionValueCoefficient = .5			# v loss coefficient
        self.LossFunctionEntropy = .01 	# entropy coefficient

        # Gamma - step reward discount 
        self.FutureRewardDiscount = 0.99

        # How many steps is used to calculacte discounted return
        self.RewardCalcualtionSteps = 8

        # Discount factor for N-Step reward
        self.N_StepRewardDiscount = self.FutureRewardDiscount ** self.RewardCalcualtionSteps # ** is power operator (Gamma_N = Math.Pow(Gamma, N_STEP_RETURN))

        # Exploration rate calculation values.
        self.ExplorationStart = 0.4
        self.ExplorationStop  = .15
        self.ExplorationDecreaseSteps = 75000

        # How long are we running for
        self.TrainingTime = 24 * 60 * 60 # 12h

        # How many concurrent environments.
        self.ConcurrentThreadsCount = 16

        # How many concurrent optimizers
        self.OptimizersCount = 6

        # CPU Thread switch sleep delay (1 ms?)
        self.CPU_ThreadSwitchDelay = 0.001