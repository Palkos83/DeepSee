import random
import numpy as np

class Agent:
    def __init__(self, id, config):
        self.ID = id
        self.Config = config
        self.Frames = 0
        self.memory = []	# used for n_step return
        self.R = 0.

		# Calculation of the curren exploration rate.
    def GetExplorationRate(self):
        if(self.Frames >= self.Config.ExplorationDecreaseSteps):
            return self.Config.ExplorationStop
        else:
            return self.Config.ExplorationStart + self.Frames * (self.Config.ExplorationStop - self.Config.ExplorationStart) / self.Config.ExplorationDecreaseSteps	# linearly interpolate

    def Act(self, brain, s):
        explorationRate = self.GetExplorationRate()			
        self.Frames = self.Frames + 1

        # Check if we are under the expliration threshold - if so, simply randomly pick action (do not use brain)
        if random.random() < explorationRate:
            return random.randint(0, self.Config.ActionsCount-1)

        else:
            s = np.array([s])
            p = brain.predict_p(s)[0]

            # Two policies for picking up the action - random or MAX.
            # a = np.argmax(p)

            # Generates a random sample from a given 1-D array
            # NUM_ACTIONS = Range for the random selection
            # p = propability distribution of the selection.
            a = np.random.choice(self.Config.ActionsCount, p=p[0,0])

            return a
    
    def Train(self, brain, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _  = memory[0]     # Current state is the first memory
            _, _, _, s_ = memory[n-1]   # Final state is the last memory (as we are using N-Step approach)
    
            return s, a, self.R, s_

        a_cats = np.zeros(self.Config.ActionsCount)	# turn action into one-hot representation
        a_cats[a] = 1 

        # Store the memory (Up to N memories for N-Step discounted reward calculation)
        self.memory.append( (s, a_cats, r, s_) )

        # We calculate new value of reward based on the previous memories + new one.
        self.R = ( self.R + r * self.Config.N_StepRewardDiscount ) / self.Config.FutureRewardDiscount

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)        # Get the length of the memory queue

                s, a, r, s_ = get_sample(self.memory, n) # Get the N-Step memory sequence. S0, A, R, SN

                brain.train_push(s, a, r, s_)   # Train the brain

                self.R = ( self.R - self.memory[0][2] ) / self.Config.FutureRewardDiscount
                self.memory.pop(0)		

            self.R = 0

        # If we have enough memories we can commence the training.
        if len(self.memory) >= self.Config.RewardCalcualtionSteps:
            s, a, r, s_ = get_sample(self.memory, self.Config.RewardCalcualtionSteps)
            brain.train_push(s, a, r, s_)

            # Reduce the accumulated reward by the reward from the olders memory
            self.R = self.R - self.memory[0][2]

            # Remove the oldest memory
            self.memory.pop(0)	

        # TEMP SOLUTION TO DEBUG IN SINGLE THREAD
        #brain.Optimize(self.ID)
    # possible edge case - if an episode ends in <N steps, the computation is incorrect

#---------
