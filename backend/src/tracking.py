import numpy as np

class SimpleTracker:
    def __init__(self):
        self.prev_centroids = []

    def track(self, boxes):
        centroids = []
        for b in boxes:
            c = np.mean(b, axis=0)
            centroids.append(c)
        self.prev_centroids = centroids
        return centroids