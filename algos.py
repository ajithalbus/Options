import gym
import numpy as np
import sys
import pygame 
import math
import argparse
import matplotlib.pyplot as plt

def showPolicy(q,mainGoal):
	BLACK = (0, 0, 0)
	WHITE = (255, 255, 255)
	GREEN = (0, 255, 0)
	RED1 = (160, 69, 46)
	RED2 = (183, 52, 20)
	RED3 = (255, 50, 0)
	RED = (255,0,0)
	BLUE = (0,0,255)
	WIDTH = 20
	HEIGHT = 20

	MARGIN = 5

	grid1 = np.argmax(q,2)
	grid= np.loadtxt('./Env/envs/map')


	pygame.init()

	WINDOW_SIZE = [325, 325]
	screen = pygame.display.set_mode(WINDOW_SIZE)

	pygame.display.set_caption("Learnt Policy Grid")


	clock = pygame.time.Clock()

	done = False

	mysurf=pygame.Surface((15,15))
	mysurf.fill(WHITE)
	pygame.draw.polygon(mysurf, BLACK, ((0, 5), (0, 10), (10, 10), (10, 15), (15, 7), (10, 0), (10, 5)))
	mysurf=pygame.transform.rotate(mysurf,180)
	
	redSurf=pygame.Surface((15,15))
	redSurf.fill(RED)
	blueSurf=pygame.Surface((15,15))
	blueSurf.fill(BLUE)
	paints=[redSurf,blueSurf]
	while not done:
		for event in pygame.event.get():  # User did something
			if event.type == pygame.QUIT:  # If user clicked close
				done = True  # Flag that we are done so we exit this loop
		
		for row in range(13):
			for column in range(13):
				color = WHITE
				ifHALL=((row,column) in [(3,6),(7,9),(10,6)])*1
				if grid1[row,column]==0:
					arrow=paints[0-ifHALL]
				elif grid1[row,column]==1:
					arrow=paints[1-ifHALL]
				else:
					arrow=pygame.transform.rotate(mysurf,-90*(grid1[row,column]-2))
				if grid[row][column] == 1:
					color = RED1
				elif (row,column) == mainGoal:
					color = GREEN
				elif grid[row,column] == -3:
					color = RED3
				elif grid[row,column] > 0:
					#print row,column
					color = GREEN
				pygame.draw.rect(screen,
								color,
								[(MARGIN + WIDTH) * column + MARGIN,
								(MARGIN + HEIGHT) * row + MARGIN,
								WIDTH,
								HEIGHT])
				if grid[row,column]!=1 and (row,column) != mainGoal:
					screen.blit(arrow,((MARGIN + WIDTH) * column + MARGIN,
								(MARGIN + HEIGHT) * row + MARGIN))
		
		clock.tick(60)

		pygame.display.flip()
	pygame.quit()

def epsilonChoice(epsilon,q):
	policy=np.zeros(4)
	action = np.random.choice(np.flatnonzero(q == q.max()))
	for i in range(4): 
		policy[i]=epsilon/4
		policy[action]=1-epsilon+epsilon/4
	action=np.random.choice(4,p=policy)
	return action	
	

def runQGame(env,option,episodes=1000,alpha=0.001,epsilon=0.1,gamma=0.9,showPolicyAtEnd=False,render=False,verbose=False,lam=None):
	Q = np.zeros([env.observation_space.spaces[0].n,env.observation_space.spaces[1].n, env.action_space.n])
	#goal,startx,starty,addStart=optionParams
	AllSteps=[]
	AllRewards=[]
	for episode in range(episodes):
		done = False
		G=0
		
		state_t = env.reset(option.Initiation,goal=option.Goal,Gui=render)
		steps=0
		
		while not done:
			steps+=1
			if steps>1000:
				break
			#epsilon-greedy
			action = epsilonChoice(epsilon,Q[state_t[0],state_t[1]])
			
			state_tp1, reward, done,_ = env.step(action) 
			if episode>(episodes-episodes/10) and render:
				env.render()
			Q[state_t[0],state_t[1],action] += alpha * (reward + gamma*np.max(Q[state_tp1[0],state_tp1[1]]) - Q[state_t[0],state_t[1],action]) 
			G += reward
			state_t = state_tp1
			if (state_t[0],state_t[1]) in option.Termination:
				break
		AllSteps.append(steps)
		AllRewards.append(G)   
		if episode % (episodes/10) == 0 and verbose==True:
			print 'Episode: {} Return : {} Steps: {}'.format(episode,G,steps)
	if showPolicyAtEnd:
		showPolicy(Q,option.Goal)
	option.setQ(Q)
	return AllSteps,AllRewards,Q

def runDefaultQGame(env,episodes=1000,alpha=0.5,epsilon=0.1,gamma=0.9,showPolicyAtEnd=False,render=False,verbose=False,lam=None,mainGoal=(7,9)):
	Q = np.zeros([env.observation_space.spaces[0].n,env.observation_space.spaces[1].n, env.action_space.n])
	
	AllSteps=[]
	AllRewards=[]
	for episode in range(episodes):
		done = False
		G=0
		
		state_t = env.reset(goal=mainGoal,Gui=render)
		steps=0
		
		while not done:
			steps+=1
			if steps>1000:
				break
			#epsilon-greedy
			action = epsilonChoice(epsilon,Q[state_t[0],state_t[1]])
			
			state_tp1, reward, done,_ = env.step(action) 
			if episode>(episodes-episodes/10) and render:
				env.render()
			Q[state_t[0],state_t[1],action] += alpha * (reward + gamma*np.max(Q[state_tp1[0],state_tp1[1]]) - Q[state_t[0],state_t[1],action]) 
			G += reward
			state_t = state_tp1
		AllSteps.append(steps)
		AllRewards.append(G)   
		if episode % (episodes/10) == 0 and verbose==True:
			print 'Episode: {} Return : {} Steps: {}'.format(episode,G,steps)
	if showPolicyAtEnd:
		showPolicy(Q,mainGoal)
	return AllSteps,AllRewards
