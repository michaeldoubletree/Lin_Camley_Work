# Michael Lin
# Only MSD

import matplotlib.pyplot as plt # imports appropriate plotting package
import math
import numpy as np
import pandas as pd
from scipy.integrate import odeint

# Python Code to find approximation of a ordinary differential equation 
# using euler method. 

dTheta = 1 # diffusion coefficient
def func_theta(): # differential equation for theta
    return (math.sqrt(2 * dTheta) * np.random.randn())

v0 = 1 # initial velocity
k = 2 # force constant
def funcx_r(theta, posit_x): # differential equation for x-component of position
    return(v0 * math.cos(theta) - k * posit_x) 
def funcy_r(theta, posit_y): # differential equation for y-component of position
    return(v0 * math.sin(theta) - k * posit_y)
      
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
        position = position + stepsize * fun(temp_theta, position)  

n = 1000 # number of times that program will iterate
h = 0.05 # timestep
tmax = 10.05 # endpoint t - h 
colnum = int(tmax / h)
all_r = np.empty([n, colnum, 2], dtype = float)

for i in range(0, n):
    #########################################################
    # FINDING THETAS #
    #########################################################

    t0 = 0 # initial time
    theta0 = 2 * math.pi * np.random.randn() # initial theta; consider adding 2 pi times randn
    theta_list = np.empty(colnum) # list of all thetas
    t = np.linspace(0, 10, 201)


    euler_theta(t0, theta0, h, tmax, theta_list, func_theta) # stored in theta_list

    #########################################################
    # FINDING POSITION #
    #########################################################

    r_x0 = 0 # initial x-coordinate of position
    r_y0 = 0 # initial y-coordinate of position
    r_xlist = np.empty(colnum) # list of all x-coordinates of position
    r_ylist = np.empty(colnum) # list of all y-coordinates of position

    euler_r(t0, r_x0, h, tmax, all_r[i,:,0], theta_list, funcx_r) # x-coordinates
    euler_r(t0, r_y0, h, tmax, all_r[i,:,1], theta_list, funcy_r) # y-coordinates



########################################
# FINDING MEAN SQUARED DISPLACEMENT #
########################################
calc_msd = np.empty(colnum) # list of calculated msd from t = 0 to t = tmax at intervals of stepsize h


# FINDING EXPECTED MSD AT EACH TIME STAMP 
expec_msd = 2 * (v0**2 / dTheta**2) * (np.exp(-dTheta * t) + dTheta * t - np.ones(np.size(t)))

# FINDING THE ACTUAL MSD AT EACH TIME STAMP # t = 0.6 (j = 12), 9.45 always inf
for j in np.arange(np.size(all_r[0,:,0])):
    diffx = all_r[:,j,0] - all_r[:,0,0]
    diffy = all_r[:,j,1] - all_r[:,0,1]
    temp = diffx**2 + diffy**2
    temp /= np.size(all_r[:,0,0])
    calc_msd[j] = temp.sum()


########################################
# Saving results into Excel Sheet
########################################

print('Saving data...')
df = pd.DataFrame({'Time': t, 'Expected MSD': list(expec_msd), 'Calculated MSD': list(calc_msd)})
writer = pd.ExcelWriter('../MSD Results/msd_springforce.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='MSD results Spring Force', index=False)
writer.save()
print('########################################')
print('Saved.')

plt.loglog(t, expec_msd, label = 'Expected MSD')
plt.loglog(t, calc_msd, label = 'Calculated MSD')
plt.axvline(x = (1/dTheta), linestyle = '--', label = '1/D')
plt.legend()
plt.show()