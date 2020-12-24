import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

# Cost Function
def cross_in_tray(x,y):
	bracket = np.abs(np.sin(x)*np.sin(y)*np.exp(np.abs(100-(np.sqrt(x**2+y**2)/np.pi))))+1
	return -0.0001*np.power(bracket,0.1)

# Plot the contour
def plot_cost_function(pos,num):
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter(pos[0],pos[1],cross_in_tray(pos[0],pos[1]),c='g',label='gbest')

	X = np.arange(-10,10,0.1)
	Y = np.arange(-10,10,0.1)
	X,Y = np.meshgrid(X,Y)

	surf = ax.plot_surface(X,Y,cross_in_tray(X,Y),cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.15)
	# plt.show()
	# fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.xlabel('x')
	plt.xticks([-10,-5,0,5,10])
	plt.yticks([-10,-5,0,5,10])
	plt.ylabel('y')
	plt.legend()
	plt.title('gbest at Iteration: {}'.format(num+1))
	plt.savefig('images/{}.png'.format(num+1))
	plt.close()


class Particle(object):
	def __init__(self,position):
		self.pos = position
		self.vel = 0
		self.cost = self.compute_cost()
		self.best_pos = position
		self.best_cost = self.cost
	
	def compute_cost(self):
		return cross_in_tray(self.pos[0],self.pos[1])

	def update_best_pos(self,position):
		self.best_pos = position
		self.best_cost = cross_in_tray(self.best_pos[0],self.best_pos[1])

# PSO initialization
var_min = -10
var_max = 10

max_iter = 200
pop_size = 20
w = 1 # Inertia Coefficient
w_damp = 0.99 # Damping coefficient
c1 = 2 # Personal Acceleration Coefficient
c2 = 2 # Social Acceleration Coefficient

particle = [0]*pop_size # Initializing particles array

global_best = Particle(np.array([np.random.uniform(-10,10),np.random.uniform(-10,10)]))
global_best.cost = np.inf

for i in range(pop_size):
	particle[i] = Particle(np.array([np.random.uniform(-10,10),np.random.uniform(-10,10)]))
	if(particle[i].best_cost<global_best.cost):
		global_best.pos = particle[i].pos
		global_best.cost = global_best.compute_cost()

best_costs = [0]*max_iter
best_positions = [0]*max_iter


# PSO iterations
for i in range(max_iter):
	for j in range(pop_size):
		new_vel = w*particle[j].vel	+ c1*np.random.rand(2)*(particle[j].best_pos - particle[j].pos) + c2*np.random.rand(2)*(global_best.pos - particle[j].pos)

		new_pos = particle[j].pos + new_vel
		if(abs(particle[j].pos[0])<=10 and abs(particle[j].pos[1])<=10):
			particle[j].vel = new_vel
			particle[j].pos = new_pos
			particle[j].cost = particle[j].compute_cost()

			if(particle[j].cost<particle[j].best_cost):
				particle[j].update_best_pos(particle[j].pos)

				if(particle[j].best_cost<global_best.cost):
					global_best.pos = particle[j].best_pos
					global_best.cost = global_best.compute_cost()
	
	best_costs[i] = global_best.cost
	best_positions[i] = global_best.pos
	print('Position: {0}. Cost: {1}'.format(global_best.pos,global_best.cost))
	w = w * w_damp

def plot_gbest_vs_iter(gbest):
	plt.plot(np.arange(0,len(gbest)),gbest)
	plt.title('gbest v/s iterations')
	plt.xlabel('Iterations')
	plt.ylabel('Cost of gbest')
	plt.grid()
	# plt.show()
	plt.savefig('gbest_vs_iter.png')
	plt.close()

plot_gbest_vs_iter(best_costs)
for i in range(len(best_positions)):
	plot_cost_function(best_positions[i],i)
# Uncomment below line to generate plot between best cost at each iteration vs the iterations
		
# plot_best_particle(best_positions[0])