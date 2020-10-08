# Michael Lin
# Plots positions 

import matplotlib.pyplot as plt # imports appropriate plotting package
import math
import numpy as np
import pandas as pd
import itertools

class Obstacle: # Creates class of obstacles
    def __init__(self, radius, spacing, xlim, ylim): 
        self.radius = radius # radius of obstacle (if circle)
        self.spacing = spacing # the amount of space between obstacles
        self.xstart = xlim[0] # x-coordinate where obstacles should start 
        self.xend = xlim[1] # x-coordinate where obstacles should end 
        self.ystart = ylim[0] # y-coordinate where obstacles should start 
        self.yend = ylim[1] # y-coordinate where obstacles should end


    def set_centers(self): # sets center of objects
        self.x_centers = []
        self.y_centers = []
        temp_xcoord = self.xstart
        temp_ycoord = self.ystart
        #counter = 0
        while temp_xcoord <= self.xend:
            self.x_centers.append(temp_xcoord)
            temp_xcoord += self.spacing

        while temp_ycoord <= self.yend:
            self.y_centers.append(temp_ycoord)
            temp_ycoord += self.spacing    

        #self.centers = np.empty((2, len(self.x_centers)*len(self.y_centers)), float)
        #for i in range(len(self.x_centers)):
            #for j in range(len(self.y_centers)):
                #self.centers[0][counter] = self.x_centers[i]
                #self.centers[1][counter] = self.y_centers[j]
                #counter += 1
        #self.centers = list(itertools.permutations(self.x_centers, r = 2))
        temp = itertools.product(self.x_centers, self.y_centers)
        self.centers = np.transpose(np.array(list(temp)))
    
    def focus_range(self, x, y):
        i = np.searchsorted(self.x_centers, x)
        j = np.searchsorted(self.y_centers, y)
        '''if (i-1) < 0:
            interest_x = np.linspace(self.x_centers[i], self.x_centers[i+1], 2)
        else:
            if (i+1) >= len(self.x_centers):
                interest_x = np.linspace(self.x_centers[i-1], self.x_centers[i], 2)
            else:
                interest_x = np.linspace(self.x_centers[i-1], self.x_centers[i+1], 3)'''
        interest_x = np.linspace(self.x_centers[i-1], self.x_centers[i+1], 3) # selects only 3 x's
        interest_y = np.linspace(self.y_centers[j-1], self.y_centers[j+1], 3) # selects only 3 y's
        '''if (j-1) < 0:
            interest_y = np.linspace(self.y_centers[j], self.y_centers[j+1], 2)
        else:
            if (j+1) >= len(self.y_centers):
                interest_y = np.linspace(self.y_centers[j-1], self.y_centers[j], 2)
            else:
                interest_y = np.linspace(self.y_centers[j-1], self.y_centers[j+1], 3)'''
        temp = itertools.product(interest_x, interest_y)
        interest_centers = np.transpose(np.array(list(temp)))
        return interest_centers

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

    def check_distances(self, x, y): # checks if a given point is touching/inside a circle
        close = self.focus_range(x, y)
        displacement = ((close[0] - x)**2 + (close[1] - y)**2)**0.5
        condition = np.any(displacement <= self.radius)
        return condition

    def get_normal(self, x, y): # gets normal vector from x and y coordinate of circle
        close = self.focus_range(x, y)
        displacement = ((close[0] - x)**2 + (close[1] - y)**2)**0.5
        coordinate = np.where(displacement <= self.radius)[0]
        x_circle = close[0][coordinate][0]
        y_circle = close[1][coordinate][0]
        xdiff = x - x_circle
        ydiff = y - y_circle
        r_length = math.sqrt(xdiff**2 + ydiff**2)
        normal = np.array([xdiff/r_length, ydiff/r_length])
        return normal

    def check_initial(self, r): #makes sure initial position is not in a circle and repositions accordingly
        close = self.focus_range(r[0], r[1])
        displacement = ((close[0] - r[0])**2 + (close[1] - r[1])**2)**0.5
        if np.any(displacement <= self.radius):
            coordinate = np.where(displacement <= self.radius)[0]
            x_circle = self.centers[0][coordinate][0]
            y_circle = self.centers[1][coordinate][0]
            angle = 2 * math.pi * np.random.randn()
            x_circle = self.radius * math.cos(angle) + x_circle
            y_circle = self.radius * math.sin(angle) + y_circle
            r = np.array([x_circle, y_circle])
            return r
        else:
            return r


# Python Code to find approximation of a ordinary differential equation 
# using euler method. 

dTheta = 1 # diffusion coefficient
def func_theta(): # differential equation for theta
    return (math.sqrt(2 * dTheta) * np.random.randn())

v0 = 1 # initial velocity
k = 2 # force constant
def func_r(theta, x, y, obst): # differential equation for x-component of position
    p = np.array([math.cos(theta), math.sin(theta)])
    if obst.check_distances(x, y):
        n = obst.get_normal(x, y)
        if np.sum(n*p) < 0:
            return (v0 * p + -v0 * (np.sum(n*p)) * n)
        else:
            return (v0*p)
    else:
        return (v0 * p)

      
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


def euler_r(position, t, l1, supp_l, fun1, obst): 
    # Iterating till the point at which we 
    # need approximation 
    stepsize = t[1] - t[0]
    for i in range(len(t)):
        l1[i][0] = position[0]
        l1[i][1] = position[1]
        temp_theta = supp_l[i]
        position = position + stepsize * fun1(temp_theta, position[0], position[1], obst)   

n = 3 # number of times that program will iterate
t = np.linspace(0, 100, 20001)
colnum = len(t) # of time points
all_r = np.empty([n, colnum, 2], dtype = float)
obstacle_radius = 0.5
space = 2 # space between circle centers
xlim = [-50, 50]
ylim = [-50, 50]

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

    r_0 = obstacles.check_initial(np.array([math.cos(theta0), math.sin(theta0)]))

    euler_r(r_0, t, all_r[i,:,:], theta_list, func_r, obstacles)

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

    obstacles.plot_circles()
    plt.axis('equal')
    plt.plot(all_r[i,:,0], all_r[i,:,1], 'b')
    plt.xlabel('X-Position')
    plt.ylabel('Y-Position')
    plt.title('Overall Position of Particle: Trial ' + str((i+1)))

    plt.show()

calc_msd = np.empty(colnum)
for j in np.arange(np.size(all_r[0,:,0])):
    diffx = all_r[:,j,0] - all_r[:,0,0]
    diffy = all_r[:,j,1] - all_r[:,0,1]
    temp = diffx**2 + diffy**2
    temp /= np.size(all_r[:,0,0])
    calc_msd[j] = temp.sum()

plt.loglog(t, calc_msd)
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Displacement')
plt.title('MSD v. Time')


#msd_fit = np.polyfit(t, calc_msd, 1)  # perform linear regression
#plt.loglog(t, msd_fit[0]*t+msd_fit[1], color='red')
#plt.show()
# QUestions: 
# Tau?
# incorporation of more forces? 
# edit the Obstacle class? spacing between obstacles
# Fokker Planck equation - tool for describing eventual distribution
# takes model and gives a PDE model; prob distribution of eventual positions and velocities

