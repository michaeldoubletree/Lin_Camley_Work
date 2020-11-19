# Michael Lin
# Plots positions 

import matplotlib.pyplot as plt # imports appropriate plotting package
import math
import numpy as np
import pandas as pd
import itertools

class Obstacle: # Creates class of obstacles
    def __init__(self, radius, spacing, x, y): # (self, radius, spacing, xlim, ylim)
        self.radius = radius # radius of obstacle (if circle)
        self.spacing = spacing # the amount of space between obstacles
        self.xcenter = x
        self.ycenter = y


    def periodic_method(self,x,y):
        dx = x - self.xcenter
        dy = y - self.ycenter
        dx_close = dx - (self.spacing) * np.around(dx/(self.spacing))
        dy_close = dy - (self.spacing) * np.around(dy/(self.spacing))
        return [dx_close, dy_close]


    def check_distances(self, x, y): # checks if a given point is touching/inside a circle
        close = self.periodic_method(x,y)
        #displacement = ((close[0] - x)**2 + (close[1] - y)**2)**0.5
        displacement = ((close[0])**2 + (close[1])**2)**0.5
        #condition = np.any(displacement <= self.radius)
        condition = (displacement <= self.radius)
        return condition

    def get_normal(self, x, y): # gets normal vector from x and y coordinate of circle
        close = self.periodic_method(x,y)
        r_length = math.sqrt(close[0]**2 + close[1]**2)
        normal = np.array([close[0]/r_length, close[1]/r_length])
        return normal

    def check_initial(self, r): #makes sure initial position is not in a circle and repositions accordingly
        close = self.periodic_method(r[0],r[1])
        displacement = ((close[0])**2 + (close[1])**2)**0.5

        if displacement <= self.radius:
            angle = 2 * math.pi * np.random.randn()
            x_circle = self.xcenter
            y_circle = self.ycenter
            x_circle = self.radius * math.cos(angle) + x_circle
            y_circle = self.radius * math.sin(angle) + y_circle
            r = np.array([x_circle, y_circle])
            return r
        else: 
            return r 


# Python Code to find approximation of a ordinary differential equation 
# using euler method. 

dTheta = 1 # diffusion coefficient
tau = 2 # 
def func_theta(): # differential equation for theta
    return (math.sqrt(2 * dTheta) * np.random.randn())
    #return 1/tau * math.asin() + np.random.randn()

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

def set_obstaclebound(x,y):
    minx = np.around(np.amin(x))
    miny = np.around(np.amin(y))
    maxx = np.around(np.amax(x))
    maxy = np.around(np.amax(y))
    xlim = [minx, maxx]
    ylim = [miny, maxy]
    return xlim, ylim

def plot_circles(xcenter, ycenter, xlim, ylim,spacing,radius,graphx,graphy):
    if (xcenter + xlim[0])%spacing != 0:
        xlim[0] = xlim[0] + 1
    if (xcenter + xlim[1])%spacing != 0:
        xlim[1] = xlim[1] + 1
    if (ycenter + ylim[0])%spacing != 0:
        ylim[0] = ylim[0] + 1
    if (ycenter + ylim[1])%spacing != 0:
        ylim[1] = ylim[1] + 1
    x_centers = np.arange(xlim[0],xlim[1],spacing) 
    y_centers = np.arange(ylim[0],ylim[1],spacing) 
    
    temp = itertools.product(list(x_centers), list(y_centers))
    centers = np.transpose(np.array(list(temp)))
    
    fig, ax = plt.subplots()
    for i in range(len(centers[0])):
        circle = plt.Circle((centers[0][i], centers[1][i]), radius, alpha=0.5)
        ax.add_artist(circle)
        ax.set_xlim(graphx)
        ax.set_ylim(graphy)

    
    

        

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
        if obst.check_distances(l1[i][0], l1[i][1]):
            disp = math.sqrt((position[0] - l1[i][0])**2 + (position[1] - l1[i][1])**2)
            xangle = (position[0] - l1[i][0])/disp
            yangle = (position[1] - l1[i][1])/disp
            l1[i][2] = math.atan2(yangle,xangle)
        else:
            l1[i][2] = supp_l[i]   

n = 3 # number of times that program will iterate
t = np.linspace(0, 10, 201)
colnum = len(t) # of time points
all_r = np.empty([colnum, 3], dtype = float)
obstacle_radius = 0.5
space = 2 # space between circle centers
xcenter = 0
ycenter = 0 

theta0 = 2 * math.pi * np.random.randn() # initial theta; consider adding 2 pi times randn
theta_list = np.empty(colnum) # list of all thetas

euler_theta(theta0, t, theta_list, func_theta) # stored in theta_list

#########################################################
# FINDING POSITION #
#########################################################

# sets up obstacle course
obstacles = Obstacle(obstacle_radius, space, xcenter, ycenter)
r_0 = obstacles.check_initial(np.array([math.cos(theta0), math.sin(theta0)]))
euler_r(r_0, t, all_r, theta_list, func_r, obstacles)
xlim, ylim = set_obstaclebound(all_r[:,0],all_r[:,1])
graphx = [np.amin(all_r[:,0])-1.5, np.amax(all_r[:,0]+1.5)]
graphy = [np.amin(all_r[:,1])-1.5, np.amax(all_r[:,1]+1.5)]

print(graphx)
print(graphy)

#plot_circles(xcenter, ycenter, xlim, ylim, space, obstacle_radius)
#plt.axis('equal')
#plt.plot(all_r[:,0], all_r[:,1], 'b')
#plt.xlabel('X-Position')
#plt.ylabel('Y-Position')
#plt.title('Overall Position of Particle')
#plt.show()
#plt.close()
#########################################################
# PLOT RESULTS #
#########################################################


for i in range(len(all_r[:,0])-1):
    plot_circles(xcenter, ycenter, xlim, ylim, space, obstacle_radius,graphx,graphy)
    plt.axis('equal')
    plt.plot(all_r[0:i+1,0], all_r[0:i+1,1], 'b')
    plt.quiver(all_r[i,0],all_r[i,1], math.cos(theta_list[i]), math.sin(theta_list[i]), color = 'g', label = 'Polarity')
    plt.quiver(all_r[i,0],all_r[i,1], math.cos(all_r[i,2]), math.sin(all_r[i,2]), scale = 25, color = 'r', label = 'velocity')
    plt.xlim(graphx)
    plt.ylim(graphy)
    plt.xlabel('X-Position')
    plt.ylabel('Y-Position')
    plt.title('Overall Position of Particle')
    plt.legend()
    plt.savefig('Plotting Analysis/movie/img%d.png'% i)
    plt.close()

# To Create movie, go into terminal and type
# cd Documents/GitHub/Lin_Camley_Work/Plotting\ Analysis/movie
# ffmpeg -start_number 0 -i 'img%d.png' -c:v libx264 -pix_fmt yuv420p out.mp4
