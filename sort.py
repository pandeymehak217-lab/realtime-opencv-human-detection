from __future__ import print_function
import numpy as np
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])

        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R *= 0.01
        self.kf.P *= 10
        self.kf.Q *= 0.01

        self.kf.x[:4] = np.reshape(bbox, (4,1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def update(self, bbox):
        self.time_since_update = 0
        self.kf.update(bbox)

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].flatten()

class Sort:
    def __init__(self, max_age=10, min_hits=3):
        self.trackers = []
        self.frame_count = 0
        self.max_age = max_age
        self.min_hits = min_hits

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

        matches = []
        unmatched_dets = list(range(len(dets)))
        unmatched_trks = list(range(len(trks)))

        for d, det in enumerate(dets):
            best_iou = 0
            best_t = -1
            for t, trk in enumerate(trks):
                iou_score = iou(det[:4], trk[:4])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_t = t

            if best_iou > 0.3:
                matches.append([d, best_t])
                unmatched_dets.remove(d)
                unmatched_trks.remove(best_t)

        for d, t in matches:
            self.trackers[t].update(dets[d][:4])

        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[d][:4]))

        ret = []
        for t, trk in enumerate(self.trackers):
            pos = trk.kf.x[:4].flatten()
            ret.append(np.concatenate([pos, [trk.id]]))

        return np.array(ret)


