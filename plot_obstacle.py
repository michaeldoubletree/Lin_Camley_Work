# Michael Lin
# Plots positions 

import matplotlib.pyplot as plt # imports appropriate plotting package
import math
import numpy as np
import pandas as pd
from scipy.integrate import odeint

class Obstacle: # Creates class of obstacles
    def __init__(self, radius, spacing, xlim, ylim): 
        self.radius = radius # radius of obstacle (if circle)
        self.spacing = spacing # the amount of space between obstacles
        self.xstart = xlim[0] # x-coordinate where obstacles should start 
        self.xend = xlim[1] # x-coordinate where obstacles should end 
        self.ystart = ylim[0] # y-coordinate where obstacles should start 
        self.yend = ylim[1] # y-coordinate where obstacles should end


    def set_centers(self): # sets center of objects
        x_centers = []
        y_centers = []
        temp_xcoord = self.xstart
        temp_ycoord = self.ystart
        counter = 0
        while temp_xcoord <= self.xend:
            x_centers.append(temp_xcoord)
            temp_xcoord += self.spacing

        while temp_ycoord <= self.yend:
            y_centers.append(temp_ycoord)
            temp_ycoord += self.spacing    

        self.centers = np.empty((2, len(x_centers)*len(y_centers)), float)
        for i in range(len(x_centers)):
            for j in range(len(y_centers)):
                self.centers[0][counter] = x_centers[i]
                self.centers[1][counter] = y_centers[j]
                counter += 1

    def plot_centers(self):
        plt.xlim(self.xstart, self.xend)
        plt.ylim(self.ystart, self.yend)
        plt.plot(self.centers[0], self.centers[1], 'go')
        plt.show()
    
    def plot_circles(self):
        fig, ax = plt.subplots()
        plt.xlim(self.xstart, self.xend)
        plt.ylim(self.ystart, self.yend)
        for i in range(len(self.centers[0])):
            circle = plt.Circle((self.centers[0][i], self.centers[1][i]), self.radius)
            ax.add_artist(circle)

    def check_distances(self, x, y):
        displacement = ((self.centers[0] - x)**2 + (self.centers[1] - y)**2)**0.5
        print(displacement)
        condition = np.any(displacement <= self.radius)
        return condition

        



# Python Code to find approximation of a ordinary differential equation 
# using euler method. 

dTheta = 1 # diffusion coefficient
def func_theta(): # differential equation for theta
    return (math.sqrt(2 * dTheta) * np.random.randn())

v0 = 1 # initial velocity
k = 2 # force constant
def funcx_r(theta): # differential equation for x-component of position
    return(v0 * math.cos(theta)) 
def funcy_r(theta): # differential equation for y-component of position
    return(v0 * math.sin(theta))
      
# Function for euler formula; will be only used to find the theta 
#########################################################
# t0 is initial time, theta is the angle found through each iteration of function,
# stepsize is the increment the Euler function ascends by, tmax is the upper bound of time
# l1 is the list that contains each theta value per iteration, fun is the differential 
# function 
def euler_theta(theta, t, l1, fun): 
    # Iterating till the point at which we 
    # need approximation
    stepsize = t[1] - t[0]
    for i in range(len(t)):
        l1[i] = theta
        theta = theta + math.sqrt(stepsize) * fun()


# Function for euler formula; adjusted to find the position since it requires theta
#########################################################
# t0 is initial time, position is the position found through each iteration of function,
# stepsize is the increment the Euler function ascends by, tmax is the upper bound of time
# l1 is the list that contains each position value per iteration, fun is the differential 
# function, supp_l is the supplemental list of thetas that is provided to find the
# positions

def euler_r(position, t, l1, supp_l, fun): 
    # Iterating till the point at which we 
    # need approximation 
    stepsize = t[1] - t[0]
    for i in range(len(t)):
        l1[i] = position
        temp_theta = supp_l[i]
        position = position + stepsize * fun(temp_theta)  

n = 3 # number of times that program will iterate
t = np.linspace(0, 10, 201)
colnum = len(t) # of time points
all_r = np.empty([n, colnum, 2], dtype = float)
obstacle_radius = 0.5
space = 2
xlim = [-5, 5]
ylim = [-5, 5]

a = Obstacle(obstacle_radius, space, xlim, ylim)
a.set_centers()
a.plot_centers()
a.plot_circles()
plt.plot(1.0,1.5, 'go')
plt.show()
print(a.check_distances(1.0, 1.5))


'''
for i in range(0, n):
    #########################################################
    # FINDING THETAS #
    #########################################################

    theta0 = 2 * math.pi * np.random.randn() # initial theta; consider adding 2 pi times randn
    theta_list = np.empty(colnum) # list of all thetas

    euler_theta(theta0, t, theta_list, func_theta) # stored in theta_list

    #########################################################
    # FINDING POSITION #
    #########################################################

    # sets up obstacle course
    obstacles = Obstacle(obstacle_radius, space, xlim, ylim)
    obstacles.set_centers()

    r_x0 = math.cos(theta0) # initial x-coordinate of position
    r_y0 = math.sin(theta0) # initial y-coordinate of position

    euler_r(r_x0, t, all_r[i,:,0], theta_list, funcx_r) # x-coordinates
    euler_r(r_y0, t, all_r[i,:,1], theta_list, funcy_r) # y-coordinates

    #########################################################
    # PLOT RESULTS #
    #########################################################


    # Plots Theta values

    plt.plot(t, theta_list, 'g')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Angle (Radians)')
    plt.title('Angle v. Time: Trial ' + str((i+1)))

    plt.show()

    # Plots individual x and y positions with respect to time
    plt.figure()

    plt.subplot(211)
    plt.plot(t, all_r[i,:,0], 'g')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Position (???)')
    plt.title('X-Position v. Time: Trial ' + str((i+1)))
    plt.tight_layout()

    plt.subplot(212)
    plt.plot(t, all_r[i,:,1], 'g')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Position (???)')
    plt.title('Y-Position v. Time: Trial ' + str((i+1)))
    plt.tight_layout()

    plt.show()

    # Plots both positions at the same time to identify overall position


    plt.plot(all_r[i,:,0], all_r[i,:,1], 'b')
    plt.xlabel('X-Position')
    plt.ylabel('Y-Position')
    plt.title('Overall Position of Particle: Trial ' + str((i+1)))

    plt.show()
    
    # Turns the Positions to Histograms

    plt.figure()

    plt.subplot(211)
    plt.hist(all_r[i,:,0], edgecolor = 'black', linewidth = 1)
    plt.xlabel('X-Position')
    plt.ylabel('Frequency')
    plt.title('X-Position Frequencies: Trial ' + str((i+1)))

    plt.subplot(212)
    plt.hist(all_r[i,:,1], edgecolor = 'black', linewidth = 1)
    plt.xlabel('Y-Position')
    plt.ylabel('Frequency')
    plt.title('Y-Position Frequencies: Trial ' + str((i+1)))

    plt.tight_layout()
    plt.show()

    # Histogram of overall position
    plt.figure()
    r_overall = np.sqrt(all_r[i,:,0]**2 + all_r[i,:,1]**2)
    plt.hist(r_overall, edgecolor = 'black', linewidth = 1)
    plt.xlabel('Overall Distance from Origin')
    plt.ylabel('Frequency')
    plt.title('Overall Position Frequencies: Trial ' + str((i+1)))
    plt.show()
'''