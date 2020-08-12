import matplotlib.pyplot as plt # imports appropriate plotting package
import math
import numpy as np
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
def euler_theta( x0, y, h, x, l1, fun): 
    # Iterating till the point at which we 
    # need approximation 
    while x0 < x: 
        l1.append(y)
        y = y + h * fun()
        x0 = x0 + h 

# Function for euler formula; adjusted to find the position since it requires theta
def euler_r( x0, y, h, x, l1, supp_l, fun): 
    # Iterating till the point at which we 
    # need approximation 
    while x0 < x:
        temp_element = int(x0 / h)
        temp_theta = supp_l[temp_element]
        l1.append(y)
        y = y + h * fun(temp_theta)
        x0 = x0 + h 
  
#########################################################
# FINDING THETAS #
#########################################################

t0 = 0 # initial time
theta0 = 0 # initial theta
h = 0.05 # timestep
tmax = 10.05 # endpoint t - h 
theta_list = [ ] # list of all thetas
t = np.linspace(0, 10, 201)

euler_theta(t0, theta0, h, tmax, theta_list, func_theta) # stored in theta_list

#########################################################
# FINDING POSITION #
#########################################################

r_x0 = math.cos(theta0) # initial x-coordinate of position
r_y0 = math.sin(theta0) # initial y-coordinate of position
r_xlist = [ ] # list of all x-coordinates of position
r_ylist = [ ] # list of all x-coordinates of position

euler_r(t0, r_x0, h, tmax, r_xlist, theta_list, funcx_r) # x-coordinates
euler_r(t0, r_y0, h, tmax, r_ylist, theta_list, funcy_r) # y-coordinates

#########################################################
# PLOT RESULTS #
#########################################################

# Plots Theta values

plt.plot(t, theta_list, 'g')
plt.xlabel('Time (Seconds)')
plt.ylabel('Angle (Radians)')
plt.title('Angle v. Time')

plt.show()

# Plots individual x and y positions with respect to time
plt.figure()

plt.subplot(211)
plt.plot(t, r_xlist, 'g')
plt.xlabel('Time (Seconds)')
plt.ylabel('Position (???)')
plt.title('X-Position v. Time')
plt.tight_layout()

plt.subplot(212)
plt.plot(t, r_ylist, 'g')
plt.xlabel('Time (Seconds)')
plt.ylabel('Position (???)')
plt.title('Y-Position v. Time')
plt.tight_layout()

plt.show()

# Plots both positions at the same time to identify overall position

plt.plot(r_xlist, r_ylist, 'b')
plt.xlabel('X-Position')
plt.ylabel('Y-Position')
plt.title('Overall Position of Particle')

plt.show()

# FINDING MEAN SQUARED DISPLACEMENT #
msd = 0
initX = r_xlist[0]
initY = r_ylist[0]

for i in range(len(r_xlist)):
    diffx = r_xlist[i] - initX
    diffy = r_ylist[i] - initY
    difflen = math.sqrt(diffx**2 + diffy**2)
    temp = difflen**2
    msd = msd + temp

msd = msd / len(r_xlist)

expecmsd = 2 * (v0**2 / dTheta**2) * (math.exp(-dTheta * (tmax-h)) + dTheta * (tmax-h) - 1)
print('Expected MSD: ')
print(expecmsd)
print('Calculated MSD: ')
print(msd)