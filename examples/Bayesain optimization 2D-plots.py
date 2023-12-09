#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[123]:


def objective(x, y):
    # Calculate the radial distance from the origin
    r1 = np.sqrt((x - 3)**2 + (y - 3)**2)  # Centered at (3, 3)
    r2 = np.sqrt((x + 5)**2 + (y + 5)**2)  # Centered at (-5, -5)
    
    # Create two circular patterns using a custom fractional formula
    pattern1 = np.sin(r1) / r1  # Fractional function for pattern 1
    pattern2 = np.sin(r2) / r2  # Fractional function for pattern 2
    
    # Combine the patterns from the two coordinates
    combined_pattern = pattern1 + pattern2
    
    return -combined_pattern

xlist = np.linspace(-9, 10, 100)
ylist = np.linspace(-9, 10, 100)
[X, Y] = np.meshgrid(xlist, ylist)
Z = -objective(X, Y)

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(X, Y, Z, 1000, cmap='jet')
fig.colorbar(cp)  # Add a colorbar to the plot
ax.set_title('Two Circular Patterns')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# In[124]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

# Create the surface plot
surf = ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.9)

# Hide axis labels and ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Hide the x, y, and z coordinate planes
ax.axis('off')

plt.show()


# In[76]:


from skopt import Optimizer
from skopt.plots import plot_objective
from skopt.plots import plot_evaluations


# In[111]:


opt = Optimizer([(-10.0,10.0),(-10.0,10.0)], "GP", acq_func='LCB',
                acq_optimizer="sampling",initial_point_generator="lhs",
                n_initial_points=10)

# To obtain a suggestion for the point at which to evaluate the objective
# you call the ask() method of opt:


# In[112]:


for i in range(30):
    next_x = np.round(opt.ask(),1).tolist()
    f_val = -np.round(objective(next_x[0],next_x[1]),6)
    res = opt.tell(next_x, -f_val)
        


# In[120]:


fig,ax=plt.subplots(1,1)
xlist = np.linspace(-10, 10, 100)
ylist = np.linspace(-10, 10, 100)
[X, Y] = np.meshgrid(xlist, ylist)
Z = -objective(X, Y)
cp = ax.contourf(X, Y, Z,100, cmap='jet')
ax.scatter(np.array(res.x_iters)[:,0],np.array(res.x_iters)[:,1],s=100, c='black');
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Bayesian optimization')
ax.set_xlabel('x ')
ax.set_ylabel('y ')
plt.show()



# In[19]:


from skopt.plots import plot_evaluations
_ = plot_evaluations(res, bins=10)


# In[21]:


from skopt.plots import plot_objective

_ = plot_objective(res)
plt.savefig('savefig_default.png')


# In[22]:


res

