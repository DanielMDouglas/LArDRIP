import numpy as np

class logManager:
    def __init__(self, outFile):
        self.outFile = outFile
        
        self.lossHist = []

        # TODO: handle an existing log file

    def update(self, loss):
        self.lossHist.append(loss.item())

    def write_to_disk(self):
        with open(self.outFile, 'a') as f:
            np.savetxt(f, self.lossHist)

        self.lossHist = []
