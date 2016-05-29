# Runaway Robot

This is my solution for the final project Runaway Robot of
[Artificial Intelligence for Robotics](https://www.udacity.com/course/artificial-intelligence-for-robotics--cs373)
in Udacity.

After trying several methods, I finally get **Part 4** solved. The optional **Part 5**
is really challenging as noise gets bigger. I am still working on it.

### Brief Explanation of the Method.
At first, before **Part 4**, I used a method, which keeps all measurements and calculate
the means of all turnings and distances. This method worked find until **Part 4**, where
hunter robot has the same speed as the runaway robot. It turns out that this method
cannot have a good estimation of the next position.

Then, I tried Particle Filter, which failed as well. I think this failure is partly
because I don't understand Particle Filter very well and cannot implement it specifically
for this problem. So, I may try Particle Filter again after having a better understanding
of it.

The method I use right now is Kalman Filter. At the beginning, how to apply Kalman
Filter to this problem seemed to be a big problem, because the Kalman Filter is
only available on linear systems. So, if my state vector is **[x, y, heading, turning, distance]**,
then I am unable to find a state transition matrix **F**. Finally, I decided that only
**heading**, **turning** and **distance** were needed to estimate the next position, so
I got rid of **x** and **y**. Then, it is rather easy to find **F**. Following is
the matrices used in the program.
```
# x: [[heading], [turning], [distance]]
# z: [[heading], [distance]]
x = matrix([[0.],
            [0.],
            [0.]])              # initial state (location and velocity)
P = matrix([[1000., 0., 0.],
            [0., 1000., 0.],
            [0., 0., 1000.]])   # initial uncertainty
u = matrix([[0.],
            [0.],
            [0.]])  # external motion
F = matrix([[1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])      # next state function
H = matrix([[1., 0., 0.],
            [0., 0., 1.]])      # measurement function
R = matrix([[1, 0.],
            [0., 1]])           # measurement uncertainty
I = matrix([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])      # identity matrix
```
Since the measurement noise is small, I can use estimated **heading**,
**turning** and **distance** with current measurement to predicate the next
position.

But as the measurement noise is getting larger, this method degrades quickly.
Therefore, it cannot succeed in **Part 5**.