###################################
#
#	Written by: Nitay Hason
#	Email: nitay.has@gmail.com
#
###################################
from time import time
from collections import deque

class StateVector:
    def __init__(self):
        self.dict = {}

    def add(self, id, state, duration):
        new_flag = False
        timestamp = time()
        if(id not in self.dict):
            self.dict[id] = {"states":{}}
        if(state in self.dict[id]["states"]):
            popped = self.dict[id]["states"][state].pop()
            if(timestamp-popped["last_timestamp"]<3): #smaller than 3 seconds
                popped["duration"]+=duration
                popped["last_timestamp"]=timestamp
                self.dict[id]["states"][state].append(popped)
            else:
                new_flag=True
                self.dict[id]["states"][state].append(popped)
                self.dict[id]["states"][state].append({"start_time":timestamp,"duration":duration,"last_timestamp":timestamp})
        else:
            self.dict[id]["states"][state]=deque()
            self.dict[id]["states"][state].append({"start_time":timestamp,"duration":duration,"last_timestamp":timestamp})

        return new_flag

    def popleft(self, id, state):
        return self.dict[id]["states"][state].popleft()

    def length(self,id, state):
        return len(self.dict[id]["states"][state])
