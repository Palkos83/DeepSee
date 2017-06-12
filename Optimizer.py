import time, threading

#---------
class Optimizer(threading.Thread):
    stop_signal = False
    
    def __init__(self, id, brain):
        self.Brain = brain
        self.ID = id

        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            self.Brain.Optimize(self.ID)

    def stop(self):
        self.stop_signal = True