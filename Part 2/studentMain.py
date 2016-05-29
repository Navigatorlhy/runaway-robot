# ----------
# Part Two
#
# Now we'll make the scenario a bit more realistic. Now Traxbot's
# sensor measurements are a bit noisy (though its motions are still
# completetly noise-free and it still moves in an almost-circle).
# You'll have to write a function that takes as input the next
# noisy (x, y) sensor measurement and outputs the best guess
# for the robot's next position.
#
# ----------
# YOUR JOB
#
# Complete the function estimate_next_pos. You will be considered
# correct if your estimate is within 0.01 stepsizes of Traxbot's next
# true position.
#
# ----------
# GRADING
#
# We will make repeated calls to your estimate_next_pos function. After
# each call, we will compare your estimated position to the robot's true
# position. As soon as you are within 0.01 stepsizes of the true position,
# you will be marked correct and we will tell you how many steps it took
# before your function successfully located the target bot.

# These import steps give you access to libraries which you may (or may
# not) want to use.
from robot import *  # Check the robot.py tab to see how this works.
from math import *
from matrix import *  # Check the matrix.py tab to see how this works.
import random
import copy

# This is the function you have to write. Note that measurement is a
# single (x, y) point. This function will have to be called multiple
# times before you have enough information to accurately predict the
# next position. The OTHER variable that your function returns will be
# passed back to your function the next time it is called. You can use
# this to keep track of important information over time.


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


def Gaussian(mu, sigma, x):
    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))


def predicate_particle(measurement, OTHER=None):

    if not OTHER:
        OTHER = []
        N = 1000
        for i in range(N):
            r = robot(measurement[0], measurement[1],
                      0, random.random() * 2 * pi - pi,
                      random.random() * 2)
            OTHER.append(r)

    p = OTHER
    N = len(p)
    Z = measurement

    w = []
    for i in range(N):
        w.append(1000 - distance_between(p[i].sense(), Z))

    p3 = []
    index = int(random.random() * N)
    beta = 0.0
    mw = max(w)
    for i in range(N):
        beta += random.random() * 2.0 * mw
        while beta > w[index]:
            beta -= w[index]
            index = (index + 1) % N
        r = copy.copy(p[index])
        r.x = measurement[0]
        r.y = measurement[1]
        p3.append(r)
    p = p3
    # p2 = []
    # tolerance = 0.01
    # while True:
    #     for i in xrange(N):
    #         if distance_between(p[i].sense(), Z) < tolerance:
    #             p2.append(p[i])
    #     if len(p2) > 0:
    #         break
    #     tolerance += 0.1
    #
    # p3 = []
    # for i in xrange(N):
    #     r = copy.copy(p2[i % len(p2)])
    #     r.x = measurement[0]
    #     r.y = measurement[1]
    #     r.turning += random.random() * tolerance - tolerance
    #     r.distance += random.random() * tolerance - tolerance
    #     p3.append(r)
    # p = p3

    sum_x = 0.0
    sum_y = 0.0
    for i in range(N):
        p[i].move_in_circle()
        sum_x += p[i].x
        sum_y += p[i].y

    est_pos = (sum_x / len(p), sum_y / len(p))
    OTHER = p
    return est_pos, OTHER


# u = matrix([[0.], [0.], [0.]])  # external motion
F = matrix([[1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])      # next state function
H = matrix([[1., 0., 0.],
            [0., 0., 1.]])      # measurement function
R = matrix([[0.075, 0.],
            [0., 0.075]])          # measurement uncertainty
I = matrix([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])      # identity matrix


def predicate_kalman(measurement, OTHER=None):
    if not OTHER:
        x = matrix([[0.],
                    [0.],
                    [0.]])  # initial state (location and velocity)
        P = matrix([[1000., 0., 0.],
                    [0., 1000., 0.],
                    [0., 0., 1000.]])  # initial uncertainty
        OTHER = [[], x, P]
    OTHER[0].append(measurement)
    # calculate heading and distance from previous data
    if len(OTHER[0]) == 1:
        m_heading = 0
        m_distance = 0
    else:
        p1 = (OTHER[0][-1][0], OTHER[0][-1][1])
        p2 = (OTHER[0][-2][0], OTHER[0][-2][1])
        m_distance = distance_between(p1, p2)
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        m_heading = atan2(dy, dx) % (2 * pi)
        OTHER[0].pop(0)

    x = OTHER[1]
    P = OTHER[2]
    pre_heading = x.value[0][0]
    for d in [-1, 0, 1]:
        diff = (int(pre_heading / (2 * pi)) + d) * (2 * pi)
        if abs(m_heading + diff - pre_heading) < pi:
            m_heading += diff
            break
    # measurement update
    y = matrix([[m_heading],
                [m_distance]]) - H * x
    S = H * P * H.transpose() + R
    K = P * H.transpose() * S.inverse()
    x = x + (K * y)
    P = (I - K * H) * P
    # prediction
    x = F * x
    P = F * P * F.transpose()

    OTHER[1] = x
    OTHER[2] = P

    est_heading = x.value[0][0]
    est_distance = x.value[2][0]
    est_x = measurement[0] + est_distance * cos(est_heading)
    est_y = measurement[1] + est_distance * sin(est_heading)

    return (est_x, est_y), OTHER


# A helper function you may find useful.
def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# This is here to give you a sense for how we will be running and grading
# your code. Note that the OTHER variable allows you to store any
# information that you want.
def demo_grading(estimate_next_pos_fcn, target_bot, OTHER=None):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0
    # if you haven't localized the target bot, make a guess about the next
    # position, then we move the bot and compare your guess to the true
    # next position. When you are close enough, we stop checking.
    while not localized and ctr <= 1000:
        ctr += 1
        measurement = target_bot.sense()
        position_guess, OTHER = estimate_next_pos_fcn(measurement, OTHER)
        target_bot.move_in_circle()
        true_position = (target_bot.x, target_bot.y)

        # x = OTHER[1]
        # print (angle_trunc(x.value[0][0]), x.value[1][0], x.value[2][0]), \
        #     (target_bot.heading, target_bot.turning, target_bot.distance), \
        #     position_guess, true_position

        error = distance_between(position_guess, true_position)
        if error <= distance_tolerance:
            print "You got it right! It took you ", ctr, " steps to localize."
            localized = True
        if ctr == 1000:
            print "Sorry, it took you too many steps to localize the target."
    return localized


def demo_grading_visualize(estimate_next_pos_fcn, target_bot, OTHER=None):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0
    # if you haven't localized the target bot, make a guess about the next
    # position, then we move the bot and compare your guess to the true
    # next position. When you are close enough, we stop checking.
    # For Visualization
    import turtle  # You need to run this locally to use the turtle module
    window = turtle.Screen()
    window.bgcolor('white')
    size_multiplier = 25.0  # change Size of animation
    broken_robot = turtle.Turtle()
    broken_robot.shape('turtle')
    broken_robot.color('green')
    broken_robot.resizemode('user')
    broken_robot.shapesize(0.1, 0.1, 0.1)
    measured_broken_robot = turtle.Turtle()
    measured_broken_robot.shape('circle')
    measured_broken_robot.color('red')
    measured_broken_robot.resizemode('user')
    measured_broken_robot.shapesize(0.1, 0.1, 0.1)
    prediction = turtle.Turtle()
    prediction.shape('arrow')
    prediction.color('blue')
    prediction.resizemode('user')
    prediction.shapesize(0.1, 0.1, 0.1)
    prediction.penup()
    broken_robot.penup()
    measured_broken_robot.penup()
    # End of Visualization
    while not localized and ctr <= 1000:
        ctr += 1
        measurement = target_bot.sense()
        position_guess, OTHER = estimate_next_pos_fcn(measurement, OTHER)
        target_bot.move_in_circle()
        true_position = (target_bot.x, target_bot.y)

        # x = OTHER[1]
        # print (angle_trunc(x.value[0][0]), x.value[1][0], x.value[2][0]), \
        #     (target_bot.heading, target_bot.turning, target_bot.distance), \
        #     position_guess, true_position

        error = distance_between(position_guess, true_position)
        if error <= distance_tolerance:
            print "You got it right! It took you ", ctr, " steps to localize."
            localized = True
        #if ctr == 1000:
        #    print "Sorry, it took you too many steps to localize the target."
        # More Visualization
        measured_broken_robot.setheading(target_bot.heading * 180 / pi)
        measured_broken_robot.goto(measurement[0] * size_multiplier, measurement[1] * size_multiplier - 200)
        measured_broken_robot.stamp()
        broken_robot.setheading(target_bot.heading * 180 / pi)
        broken_robot.goto(target_bot.x * size_multiplier, target_bot.y * size_multiplier - 200)
        broken_robot.stamp()
        prediction.setheading(target_bot.heading * 180 / pi)
        prediction.goto(position_guess[0] * size_multiplier, position_guess[1] * size_multiplier - 200)
        prediction.stamp()
        # End of Visualization
    return localized


# This is a demo for what a strategy could look like. This one isn't very good.
def naive_next_pos(measurement, OTHER=None):
    """This strategy records the first reported position of the target and
    assumes that eventually the target bot will eventually return to that
    position, so it always guesses that the first position will be the next."""
    if not OTHER:  # this is the first measurement
        OTHER = measurement
    xy_estimate = OTHER
    return xy_estimate, OTHER


# This is how we create a target bot. Check the robot.py file to understand
# How the robot class behaves.
test_target = robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)
measurement_noise = 0.05 * test_target.distance
test_target.set_noise(0.0, 0.0, measurement_noise)

# demo_grading_visualize(predicate_kalman, test_target)
demo_grading(predicate_kalman, test_target)
