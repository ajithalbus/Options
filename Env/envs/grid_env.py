#!/usr/local/bin/python
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pygame
import itertools as it

def moveVector(action):
	if action==0:
		vector=[0,-1]
	elif action==1:
		vector=[-1,0]
	elif action==2:
		vector=[0,1]
	elif action==3:
		vector=[1,0]
		
	return vector

class GridEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	def _makeEnv(self,goal=(7,9),goalReward=10):
		'''Creates a numpy array as the reward environment'''
		rews=np.loadtxt('./Env/envs/map',dtype=np.int)
		rews[goal]=goalReward #need to change later
		return rews

	def _take_action(self,action):
		
		
		if	np.random.choice([0,1],p=[0.666,0.334]): # p=0.1 take random move 
			possibleActions=[0,1,2,3]
			#possibleActions.remove(action)
			action=np.random.choice(possibleActions)
		if self.rews[(self.position + moveVector(action))[0],(self.position + moveVector(action))[1]]==1:
			return
		self.position += moveVector(action) #for taking action
		self.position = np.clip(self.position,0,11) # to avoid out of bounds
		
	

		

	def __init__(self):
		self.goal=(7,9)
		#self.rews=self._makeEnv(goal=self.goal)
		self.observation_space=spaces.Tuple((spaces.Discrete(13),spaces.Discrete(13)))
		self.action_space=spaces.Discrete(4)
		
		

		
	def step(self, action):
		self._take_action(action)
		reward = self.rews[self.position[0],self.position[1]]
		ob = np.copy(self.position)
		done = np.array_equal(self.position,list(self.goal)) 
		if reward!=10:
			reward=-0.01
		return ob,reward,done,{}

	def reset(self,InitiationSet=[i for i in it.product(range(1,6),range(1,6))],Gui=False,goal=(7,9)):
		self.goal=goal
		self.rews=self._makeEnv(self.goal)
		startX,startY=InitiationSet[np.random.choice(range(len(InitiationSet)))]
		
		a,b=np.random.choice(13),np.random.choice(13) 
		while self.rews[a,b]==1 or (a,b)==goal:
			a,b=np.random.choice(13),np.random.choice(13)
		startX,startY=a,b
		
		self.position=np.array([startX,startY])
		
		
		if Gui:
			pygame.init()
			
			WINDOW_SIZE = [330, 330]
			self.screen = pygame.display.set_mode(WINDOW_SIZE)
			
			pygame.display.set_caption("Grid")
			
			
			self.clock = pygame.time.Clock()
			

		return np.copy(self.position)
		
	def render(self, mode='human', close=False):
		BLACK = (0, 0, 0)
		WHITE = (255, 255, 255)
		GREEN = (0, 255, 0)
		RED1 = (160, 69, 46)
		RED2 = (183, 52, 20)
		RED3 = (255, 50, 0)
		
		WIDTH = 20
		HEIGHT = 20
		
		MARGIN = 5
		
		grid = self.rews
		
		done = False
		
		# Set the screen background
		self.screen.fill(BLACK)
		# Draw the grid
		for row in range(13):
			for column in range(13):
				color = WHITE
				if grid[row][column] == 1:
					color = RED1
				elif grid[row,column] == -2:
					color = RED2
				elif grid[row,column] == -3:
					color = RED3
				elif grid[row,column] > 0:
					#print row,column
					color = GREEN
				pygame.draw.rect(self.screen,
								color,
								[(MARGIN + WIDTH) * column + MARGIN,
								(MARGIN + HEIGHT) * row + MARGIN,
								WIDTH,
								HEIGHT])
		pygame.draw.circle(self.screen,BLACK,[(MARGIN + WIDTH) * self.position[1] + 3*MARGIN,
                              (MARGIN + HEIGHT) * self.position[0] + 3*MARGIN],5)
	
		self.clock.tick(60)
	
		pygame.display.flip()
	
		

		
