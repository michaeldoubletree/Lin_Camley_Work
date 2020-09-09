# Michael Lin
# Only calculates MSD for multiple particles; no plot so we can put in more points

import matplotlib.pyplot as plt # imports appropriate plotting package
import math
import numpy as np
import pandas as pd
from scipy.integrate import odeint

# Python Code to find approximation of a ordinary differential equation 
# using euler method. 

dTheta = 0.5
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
    while t0 < tmax: 
        l1.append(theta)
        theta = theta + math.sqrt(stepsize) * fun()
        t0 = t0 + stepsize 

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
    while t0 < tmax:
        temp_element = int(t0 / stepsize)
        temp_theta = supp_l[temp_element]
        l1.append(position)
        position = position + stepsize * fun(temp_theta)
        t0 = t0 + stepsize 

n = 1000 # number of times that program will iterate
h = 0.05 # timestep
tmax = 10.05 # endpoint t - h 
colnum = int(tmax / h)
all_r = [ ([0] * colnum) for l in range(n)] # lists both x and y coordinates

for i in range(0, n):
    #########################################################
    # FINDING THETAS #
    #########################################################

    t0 = 0 # initial time
    theta0 = 0 # initial theta; consider adding 2 pi times randn
    theta_list = [ ] # list of all thetas
    t = np.linspace(0, 10, 201)

    euler_theta(t0, theta0, h, tmax, theta_list, func_theta) # stored in theta_list

    #########################################################
    # FINDING POSITION #
    #########################################################

    r_x0 = math.cos(theta0) # initial x-coordinate of position
    r_y0 = math.sin(theta0) # initial y-coordinate of position
    r_xlist = [ ] # list of all x-coordinates of position
    r_ylist = [ ] # list of all y-coordinates of position

    euler_r(t0, r_x0, h, tmax, r_xlist, theta_list, funcx_r) # x-coordinates
    euler_r(t0, r_y0, h, tmax, r_ylist, theta_list, funcy_r) # y-coordinates

    # Adding to the list of values as coordinates
    # each row will have a different particle, and inside each row will be positions
    # from t = 0 to t = tmax in intervals of stepsize h.
    # X-coordinates are in first element, y-coordinate are in second element 
    for j in range(len(r_xlist)):
        all_r[i][j] = [r_xlist[j], r_ylist[j]]

########################################
# FINDING MEAN SQUARED DISPLACEMENT #
########################################
expec_msd = [ ] # list of expected msd from t = 0 to t = tmax at intervals of stepsize h
calc_msd = [ ] # list of calculated msd from t = 0 to t = tmax at intervals of stepsize h


# FINDING EXPECTED MSD AT EACH TIME STAMP 
for q in t:
    temp = 2 * (v0**2 / dTheta**2) * (math.exp(-dTheta * q) + dTheta * q - 1)
    expec_msd.append(temp)

# FINDING THE ACTUAL MSD AT EACH TIME STAMP
for j in range(len(all_r[0])):
    temp = 0 
    for i in range(len(all_r)): # iterates through each point in a particular time point
        initX = all_r[i][0][0] # initial x-coordinate
        initY = all_r[i][0][1] # initial y-coordinate
        diffx = all_r[i][j][0] - initX # difference between x coordinates
        diffy = all_r[i][j][1] - initY # difference between y coordinates
        difflensq = diffx**2 + diffy**2 # the length squared 
        temp = temp + difflensq # adds up all the length squared at a particular time point
    temp = temp / len(all_r) # divide by number of particles
    calc_msd.append(temp) # appends to the calculated list at a particular timestamp
        
# Making Data more viewable/user friendly #
results_display = [ ([0] * 3) for l in range(len(calc_msd))] # sets up list with 
# 1st column displaying time, 2nd column displaying expected MSD at that time,
# 3rd column displaying calculated MSD from data at that time.

for i in range(len(results_display)):
    if i == 0: # titles the data in the first row
        results_display[i][0] = 'Time'
        results_display[i][1] = 'Expected MSD'
        results_display[i][2] = 'Calculated MSD'
    else: # puts the time, expected msd, calculated msd in their respective columns
        results_display[i][0] = t[i]
        results_display[i][1] = expec_msd[i]
        results_display[i][2] = calc_msd[i]


########################################
# Saving results into Excel Sheet
########################################

print('Saving data...')
df = pd.DataFrame(results_display)
writer = pd.ExcelWriter('../MSD Results/msd_100.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='MSD results', index=False)
writer.save()
print('########################################')
print('Saved.')

plt.loglog(t, expec_msd)
plt.loglog(t, calc_msd)
plt.axvline(x = math.log(1/dTheta))
plt.show()