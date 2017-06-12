import numpy as np
import random

class TheGame:
    def __init__(self):

        self.Ground = 0
        self.Player = 10
        self.MoveCounter = 0;

        self.MaxMoves = 320
        self.FooChannel = 0
        self.RewardChannel = 1
        self.PlayerChannel = 2
        self.BoardImageScalingFactor = 4
        
        self.Size = 21
        self.Channels = 3
        self.PlayerXPos = int(self.Size / 2)
        self.PlayerYPos = int(self.Size / 2)
       
        self.SeedCount = int(self.Size * self.Size * 0.1)

        self.Board = np.zeros((self.Size, self.Size, self.Channels), dtype=np.uint8)

        self.ActionUp = 0
        self.ActionDown= 0
        self.ActonLeft = 0
        self.ActionRight= 0
        self.IsGameFinished = False

        self.PlayerColor = 255
        self.RewardPoint = 192
        self.PenaltyPoint = 64

        self.Reset()

    def Reset(self):
        self.MoveCounter = 0
        self.IsGameFinished = False

        self.Board = np.zeros((self.Size, self.Size, 1), dtype=np.uint8)
        for step in range(0, self.SeedCount):
            xpos = random.randint(0, self.Size-1)
            ypos = random.randint(0, self.Size-1)

            fieldType =  random.sample({-1, 1 }, 1)[0]
            if(fieldType ==1):
                self.Board[xpos, ypos, 0] = self.RewardPoint
            else:
                self.Board[xpos, ypos, 0] = self.PenaltyPoint

        self.PlayerXPos = int(self.Size / 2)
        self.PlayerYPos = int(self.Size / 2)
            
        self.Board[self.PlayerXPos, self.PlayerYPos, 0] = self.PlayerColor


        self.ActionUp = 0
        self.ActionDown= 0
        self.ActonLeft = 0
        self.ActionRight= 0
        
    def TakeAction(self , action):

        self.MoveCounter = self.MoveCounter +1
        if(self.MoveCounter >= self.MaxMoves):
            self.IsGameFinished = True

        newPlayerXPos = self.PlayerXPos
        newPlayerYPos = self.PlayerYPos

        # UP #
        if action == 0:
            self.ActionUp +=1
            newPlayerYPos -=1
            if newPlayerYPos <0 :
                newPlayerYPos = 0
        # DOWN #
        elif action == 1:
            self.ActionDown +=1
            newPlayerYPos +=1
            if newPlayerYPos == self.Size :
                newPlayerYPos = self.Size -1
        
        # LEFT #
        elif action == 2:
            self.ActonLeft +=1
            newPlayerXPos -=1
            if newPlayerXPos <0 :
                newPlayerXPos = 0

        # RIGTH #
        elif action == 3:
            newPlayerXPos +=1
            self.ActionRight +=1
            if newPlayerXPos == self.Size :
                newPlayerXPos = self.Size -1
        
        # Check if the position has changed
        reward =0 
        if newPlayerXPos == self.PlayerXPos and newPlayerYPos == self.PlayerYPos :
            # Penalty of 10 points for now moving
            #reward = -10
            reward = 0 
        else:
            # New position, check what is in the new position
            if self.Board[newPlayerXPos, newPlayerYPos, 0] == 0:
                reward = 0 
            else:
                
                # Check if the red channel (foo) is set to 1 or zero
                if self.Board[newPlayerXPos, newPlayerYPos, 0] == self.RewardPoint :
                    reward = 1
                else:
                    reward = -1

                # We have potencially removed an item from the board. Randomly place a new one
                newItemXPos = random.randint(0, self.Size-1)
                newItemYPos = random.randint(0, self.Size-1)
                
                while (newPlayerXPos == newItemXPos and newPlayerYPos == newItemYPos) or (self.PlayerXPos == newItemXPos and self.PlayerYPos == newItemYPos) :
                    newItemXPos = random.randint(0, self.Size-1)
                    newItemYPos = random.randint(0, self.Size-1)

                fieldType =  random.sample({-1, 1 }, 1)[0]
                if(fieldType ==1):
                    self.Board[newItemXPos, newItemYPos, 0] = self.RewardPoint
                else:
                    self.Board[newItemXPos, newItemYPos, 0] = self.PenaltyPoint

            # Update board
            self.Board[self.PlayerXPos, self.PlayerYPos, 0] = 0

            self.PlayerXPos = newPlayerXPos 
            self.PlayerYPos = newPlayerYPos 

            self.Board[self.PlayerXPos, self.PlayerYPos, 0] = self.PlayerColor

        return reward

    # Gets current state of the agent 
    def GetCurrentState(self):
        # Upscale the baord
        image = np.zeros([self.Size * self.BoardImageScalingFactor, self.Size * self.BoardImageScalingFactor, 1], dtype=np.uint8)
        
        for x in range(0, self.Size * self.BoardImageScalingFactor):
            for y in range(0, self.Size * self.BoardImageScalingFactor):
                image[x,y,0] = self.Board[int(x/self.BoardImageScalingFactor), int(y/self.BoardImageScalingFactor), 0]
           #     image[x,y,1] = self.Board[int(x/self.BoardImageScalingFactor), int(y/self.BoardImageScalingFactor), 1]
            #    image[x,y,2] = self.Board[int(x/self.BoardImageScalingFactor), int(y/self.BoardImageScalingFactor), 2]
    
     
        return image , self.IsGameFinished