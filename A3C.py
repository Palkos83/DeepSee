import numpy as np
import tensorflow as tf
from datetime import datetime

import time, random, threading
import Environment
import Brain
import Optimizer
import Config

print(datetime.now(), "Main A3C - Start")
print(datetime.now(), "Main A3C - Create config")
config = Config.Config()

print(datetime.now(), "Main A3C - Create brain instance")
#-- main
brain = Brain.Brain(config)	# brain is global in A3C

print(datetime.now(), "Main A3C - Create environemtns: ", config.ConcurrentThreadsCount)

environments = [Environment.Environment(i, brain, config) for i in range(config.ConcurrentThreadsCount)]

print(datetime.now(), "Main A3C - Create optimizers: ", config.OptimizersCount)
optimizers = [Optimizer.Optimizer(i, brain) for i in range(config.OptimizersCount)]

print(datetime.now(), "Main A3C - Start Optimizers")

# Start all optimizer threads
for optimizer in optimizers:
	optimizer.start()

print(datetime.now(), "Main A3C - Start Environments")

# Start all Environment threads
for environment in environments:
	environment.start()

print(datetime.now(), "Main A3C - Training Execution")

# Wait until they run 
time.sleep(config.TrainingTime)

print(datetime.now(), "Main A3C - Stop Environments")

# Stop all the environments
for environment in environments:
	environment.stop()

for environment in environments:
	environment.join()

print(datetime.now(), "Main A3C - Stop Optimizers")

# Stop all the optimizers
for optimizer in optimizers:
	optimizer.stop()

for optimizer in optimizers:
	optimizer.join()

# Training is now finished, time for test run.
print("Training finished")
