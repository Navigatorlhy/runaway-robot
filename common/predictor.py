from matrix import *
from utils import *


class KalmanFilter:
    # u = matrix([[0.], [0.], [0.]])  # external motion
    def __init__(self, sigma):
        self.x = matrix([[0.],
                         [0.],
                         [0.]])
        self.P = matrix([[1000., 0., 0.],
                         [0., 1000., 0.],
                         [0., 0., 1000.]])
        # measurement uncertainty
        self.R = matrix([[sigma, 0.],
                        [0., sigma]])
        # next state function
        self.F = matrix([[1., 1., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])
        # measurement function
        self.H = matrix([[1., 0., 0.],
                         [0., 0., 1.]])
        # identity matrix
        self.I = matrix([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])
        self.keep = []

    def predict(self, measurement):
        self.keep.append(measurement)
        # calculate heading and distance from previous data
        if len(self.keep) == 1:
            m_heading = 0
            m_distance = 0
        else:
            p1 = (self.keep[-1][0], self.keep[-1][1])
            p2 = (self.keep[-2][0], self.keep[-2][1])
            m_distance = distance_between(p1, p2)
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            m_heading = atan2(dy, dx) % (2 * pi)
            self.keep.pop(0)

        pre_heading = self.x.value[0][0]
        for d in [-1, 0, 1]:
            diff = (int(pre_heading / (2 * pi)) + d) * (2 * pi)
            if abs(m_heading + diff - pre_heading) < pi:
                m_heading += diff
                break
        # measurement update
        y = matrix([[m_heading],
                    [m_distance]]) - self.H * self.x
        S = self.H * self.P * self.H.transpose() + self.R
        K = self.P * self.H.transpose() * S.inverse()
        self.x = self.x + (K * y)
        self.P = (self.I - K * self.H) * self.P
        # prediction
        self.x = self.F * self.x
        self.P = self.F * self.P * self.F.transpose()

        est_heading = self.x.value[0][0]
        est_distance = self.x.value[2][0]
        est_next = next_position_in_circle(measurement[0], measurement[1],
                                           est_heading, est_distance)
        return est_next


def predicate_mean(measurement, OTHER=None):
    """Estimate the next (x, y) position of the wandering Traxbot
    based on noisy (x, y) measurements."""
    if not OTHER:
        OTHER = []
    OTHER.append(measurement)
    if len(OTHER) == 1:
        x = OTHER[0][0]
        y = OTHER[0][1]
        xy_estimate = (x, y)
    elif len(OTHER) == 2:
        x1 = OTHER[0][0]
        y1 = OTHER[0][1]
        x2 = OTHER[1][0]
        y2 = OTHER[1][1]
        dx = x2 - x1
        dy = y2 - y1
        xy_estimate = (dx + x2, dy + y2)
    else:
        headings = []
        dists = []
        edges = []
        for i in xrange(1, len(OTHER)):
            p1 = (OTHER[i][0], OTHER[i][1])
            p2 = (OTHER[i - 1][0], OTHER[i - 1][1])
            dist = distance_between(p1, p2)
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            edges.append(dy * dx)
            heading = atan2(dy, dx)
            dists.append(dist)
            headings.append(heading)

        # find turning wise
        clockwise = True
        if sum(edges) < 0:
            clockwise = False

        turnings = []
        for i in xrange(1, len(headings)):
            turning = headings[i] - headings[i - 1]
            if clockwise:
                if turning > 0:
                    turning -= 2 * pi
            else:
                if turning < 0:
                    turning += 2 * pi
            turnings.append(turning)

        est_dist = sum(dists) / len(dists)
        est_turning = sum(turnings) / len(turnings)
        est_heading = angle_trunc(headings[-1] + est_turning)
        x = OTHER[-1][0]
        y = OTHER[-1][1]
        est_x = x + est_dist * cos(est_heading)
        est_y = y + est_dist * sin(est_heading)
        xy_estimate = (est_x, est_y)

    # You must return xy_estimate (x, y), and OTHER (even if it is None)
    # in this order for grading purposes.
    return xy_estimate, OTHER
