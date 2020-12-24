import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

INPUT_MIN = -4
INPUT_MAX = 4
RES = 17
PARAMS = 39

# PSO parameters
MAX_ITER = 1500
POP_SIZE = 100
w = 1
w_damp = 0.99
c1 = 2
c2 = 2

x1 = np.linspace(INPUT_MIN,INPUT_MAX,RES)
x2 = np.linspace(INPUT_MIN,INPUT_MAX,RES)
train_data = np.array(np.meshgrid(x1,x2)).T.reshape(-1,2)
sine = np.sin((np.pi*train_data[:,0])/4)*np.sin((np.pi*train_data[:,1])/4)

def anfis(p,x1 = train_data[:,0], x2 = train_data[:,1]):

	out1 = [0]*6 # Output of layer 1
	for i in range(3):
		out1[i] = np.exp(-(((x1-p[i*2])/p[i*2+1])**2))
		out1[i+3] = np.exp(-(((x2-p[(i+3)*2])/p[(i+3)*2+1])**2))

	out2 = [0]*9
	for i in range(3):
		out2[i*3] = out1[i]*out1[3]
		out2[i*3+1] = out1[i]*out1[4]
		out2[i*3+2] = out1[i]*out1[5]

	out3 = [0]*9 # Output of layer 3
	for i in range(9):
		out3[i] = out2[i]/(out2[0]+out2[1]+out2[2]+out2[3]+out2[4]+out2[5]+out2[6]+out2[7]+out2[8])
	

	out4 = [0]*9 # Output of layer 4
	for i in range(9):
		out4[i] = out3[i]*(x1*p[12+i*3] + x2*p[13+i*3] + p[14+i*3])
	

	out5 = 0 # Output of layer 5
	out5 = np.sum(np.array(out4),axis=0)

	return out5

def cost(params):
	f1 = anfis(params)
	f2 = sine
	return np.linalg.norm(f1-f2)

class Particle(object):
	def __init__(self,position):
		self.pos = position
		self.vel = 0
		self.cost = self.compute_cost()
		self.best_pos = position
		self.best_cost = self.cost
	
	def compute_cost(self):
		return cost(self.pos)

	def update_best_pos(self,position):
		self.best_pos = position
		self.best_cost = cost(self.best_pos)

def initialize():
	out = np.zeros(PARAMS)
	out = 4*np.random.rand(PARAMS)
	out[0] *= -1
	out[4] *= -1
	return out

def isValid(params):
	a1 = params[0] in range(-4,4)
	a2 = params[4] in range(-4,4)
	a3 = params[2] in range(-4,4)
	a4 = params[6] in range(-4,4)
	a5 = params[1]>0 and params[3]>0 and params[5]>0 and params[7]>0
	out = a1 and a2 and a3 and a4 and a5
	return a5


particle = [0]*POP_SIZE # Initializing particles array

global_best = Particle(initialize())
global_best.cost = np.inf

for i in range(POP_SIZE):
	particle[i] = Particle(initialize())
	if(particle[i].best_cost<global_best.cost):
		global_best.pos = particle[i].pos
		global_best.cost = global_best.compute_cost()

best_costs = [0]*MAX_ITER
best_positions = [0]*MAX_ITER


# PSO iterations
for i in range(MAX_ITER):
	for j in range(POP_SIZE):
		new_vel = w*particle[j].vel	+ c1*np.random.rand(PARAMS)*(particle[j].best_pos - particle[j].pos) + c2*np.random.rand(PARAMS)*(global_best.pos - particle[j].pos)
		new_pos = particle[j].pos + new_vel
		if(isValid(new_pos)):
			particle[j].vel = new_vel
			particle[j].pos = new_pos
			particle[j].cost = particle[j].compute_cost()

			if(particle[j].cost<particle[j].best_cost):
				particle[j].update_best_pos(particle[j].pos)

				if(particle[j].best_cost<global_best.cost):
					global_best.pos = particle[j].best_pos
					global_best.cost = global_best.compute_cost()
					print(global_best.cost)
	
	best_costs[i] = global_best.cost
	best_positions[i] = global_best.pos
	# print('Position: {0}. Cost: {1}'.format(global_best.pos,global_best.cost))
	w = w * w_damp
	if(i%20==0):
		xx,yy = np.meshgrid(x1,x2)
		z = anfis(global_best.pos,np.ravel(xx),np.ravel(yy))
		z = z.reshape(xx.shape)
		fig = plt.figure()
		ax = fig.add_subplot(111,projection='3d')
		ax.plot_surface(xx,yy,z)
		ax.set_xlabel("x1")
		ax.set_ylabel("x2")
		ax.set_zlabel("y")
		plt.title("Iteration = {}".format(i))
		plt.savefig("{}".format(i)+".png")
		plt.close()

print(global_best.pos)