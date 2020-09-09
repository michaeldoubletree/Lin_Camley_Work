# Michael Lin
# Plots and calculates MSD for multiple particles

import matplotlib.pyplot as plt # imports appropriate plotting package
import math
import numpy as np
import pandas as pd
from scipy.integrate import odeint

# Python Code to find approximation of a ordinary differential equation 
# using euler method. 

dTheta = 1
def func_theta(): # differential equation for theta
    return (math.sqrt(2 * dTheta) * np.random.randn())

v0 = 1
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
def euler_theta( t0, theta, stepsize, tmax, l1, fun): 
    # Iterating till the point at which we 
    # need approximation
    max = int(tmax / stepsize)
    for i in range(max):
        l1[i] = theta
        theta = theta + math.sqrt(stepsize) * fun()

# Function for euler formula; adjusted to find the position since it requires theta
#########################################################
# t0 is initial time, position is the position found through each iteration of function,
# stepsize is the increment the Euler function ascends by, tmax is the upper bound of time
# l1 is the list that contains each position value per iteration, fun is the differential 
# function, supp_l is the supplemental list of thetas that is provided to find the
# positions
def euler_r( t0, position, stepsize, tmax, l1, supp_l, fun): 
    # Iterating till the point at which we 
    # need approximation 
    max = int(tmax / stepsize)
    for i in range(max):
        l1[i] = position
        temp_theta = supp_l[i]
        position = position + stepsize * fun(temp_theta)  

n = 10 # number of times that program will iterate
h = 0.05 # timestep
tmax = 10.05 # endpoint t - h 
colnum = int(tmax / h)
all_r = np.empty([n, colnum, 2], dtype = float)

for i in range(0, n):
    #########################################################
    # FINDING THETAS #
    #########################################################

    t0 = 0 # initial time
    theta0 = 2 * math.pi * np.random.randn() # initial theta
    theta_list = np.empty(colnum)
    t = np.linspace(0, 10, 201)

    euler_theta(t0, theta0, h, tmax, theta_list, func_theta) # stored in theta_list

    #########################################################
    # FINDING POSITION #
    #########################################################

    r_x0 = 0 # initial x-coordinate of position
    r_y0 = 0 # initial y-coordinate of position

    euler_r(t0, r_x0, h, tmax, all_r[i,:,0], theta_list, funcx_r) # x-coordinates
    euler_r(t0, r_y0, h, tmax, all_r[i,:,1], theta_list, funcy_r) # y-coordinates

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
