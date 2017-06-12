import TheGame
import Agent
import time, threading
import tensorflow as tf
from datetime import datetime
import png
from scipy.misc import imsave
from array2gif import write_gif
import numpy as np

class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, id, brain, config):
        threading.Thread.__init__(self)

        self.Step = 0
        self.EpisodeCount = 0
        self.ID = id
        self.Config = config
        self.Brain = brain
        self.Environment = TheGame.TheGame()
        self.Agent = Agent.Agent(id, config)

    def RunEpisode(self):
        
        print(datetime.now(), "ID=", self.ID, "New episode start")

        # Reset gym environment.
        self.Environment.Reset()
        state, isGameFinished = self.Environment.GetCurrentState()

        imagesRecording = []
        self.EpisodeCount = self.EpisodeCount +1
        
        # Zero the total reward
        accumulatedReward = 0
        while True:  
            
            self.Step = self.Step + 1
                   
            # Get some sleep so other threads can get access to CPU
            time.sleep(self.Config.CPU_ThreadSwitchDelay) # yield 

            # Get the action from the agent
            action = self.Agent.Act(self.Brain, state)

            # Get the information from the gym environment
            reward = self.Environment.TakeAction(action)

            state_ , isGameFinished = self.Environment.GetCurrentState()

            if isGameFinished: # terminal state
                state_ = None

            # Train the agent in the s,a,r -> s' transition
            self.Agent.Train(self.Brain, state, action, reward, state_)

            # Save the current state
            state = state_

            # Add to current state reward to the total reward
            accumulatedReward += reward

            # Write some information to tensorboard
            groupName = 'Agent_' + str(self.ID)
            with tf.name_scope(groupName):
                self.Brain.AddSummary(groupName + "-PlayerXPos", self.Step, self.Environment.PlayerXPos, groupName)
                self.Brain.AddSummary(groupName + "-PlayerYPos", self.Step, self.Environment.PlayerYPos, groupName)
                self.Brain.AddSummary(groupName + "-TotalReward", self.Step, accumulatedReward, groupName)
            
            # Output the state recording
            # Attemp to save as PNG file            
            #imsave(fileName, image)
            
            # If we are done - finish
            if isGameFinished or self.stop_signal:
                break

            # Add the state to the images recordings so we can save it as a gif file at the end of the episode.
         #   imagesRecording.append(np.swapaxes(state, 0, 2))

        # Print the total reward.
        print(datetime.now(), "ID=", self.ID, " Accumulated Episode Reward:", accumulatedReward)
     
        # Save the episode run as gif file        
        # write_gif(imagesRecording, gifFileName, fps=2)

    def run(self):
        while not self.stop_signal:
            self.RunEpisode()

    def stop(self):
        self.stop_signal = True
