import itertools as it
from algos import runQGame,showPolicy
import numpy as np
import sys
class Options:
    def __init__(self,id,initiation,termination,goal):
        self.id=id
        self.Initiation=initiation
        self.Termination=termination #beta
        self.Goal=goal
    def setQ(self,Q):
        self.Q=Q
        self.Policy=np.argmax(Q,2)

def epsilonChoice(epsilon,q,choices):
    policy=np.zeros(choices)
    action = np.random.choice(np.flatnonzero(q == q.max()))
    for i in range(choices): 
        policy[i]=epsilon/choices
        policy[action]=1-epsilon+epsilon/choices
    action=np.random.choice(choices,p=policy)
    return action	

def getCombs(A,B,extra=None):
    lst=[i for i in it.product(A,B)]
    if extra != None:
        lst.append(extra)
    return lst

def getHall(point,options):
    '''returns possible options for a point '''
    point=tuple(point)
    possibles=[]
    for i in range(len(options)):
        ops=options[i]
        if point in ops.Initiation:
            possibles.append(i)
    assert len(possibles)==2
    return possibles



def playOption(env,state,option,render): # should be called only when agent is in initiation of the option
    option_reward=0
    done=False
    steps=1
    state,rew,done,_=env.step(option.Policy[tuple(state)])
    option_reward+=rew
    while not done and tuple(state) not in option.Termination:
        obs,rew,done,_=env.step(option.Policy[tuple(state)])
        if render:
            env.render()
        state=obs
        steps+=1
        option_reward=rew + option_reward
    return state,option_reward,done,steps

def GetOptions(env):
    #making options
    i=getCombs(range(1,6),range(1,6),(6,2))
    t=set(getCombs(range(14),range(14)))-set(getCombs(range(1,6),range(1,6)))
    goal=(3,6)
    O1=Options('O1',i,t,goal)
    runQGame(env,option=O1)

    i=getCombs(range(1,6),range(1,6),(3,6))
    t=set(getCombs(range(14),range(14)))-set(getCombs(range(1,6),range(1,6)))
    goal=(6,2)
    O2=Options('O2',i,t,goal)
    runQGame(env,option=O2)

    i=getCombs(range(1,7),range(7,12),(3,6))
    t=set(getCombs(range(14),range(14)))-set(getCombs(range(1,7),range(7,12)))
    goal=(7,9)
    O3=Options('O3',i,t,goal)
    runQGame(env,option=O3)

    i=getCombs(range(1,7),range(7,12),(7,9))
    t=set(getCombs(range(14),range(14)))-set(getCombs(range(1,7),range(7,12)))
    goal=(3,6)
    O4=Options('O4',i,t,goal)
    runQGame(env,option=O4)

    i=getCombs(range(8,12),range(7,12),(7,9))
    t=set(getCombs(range(14),range(14)))-set(getCombs(range(8,12),range(7,12)))
    goal=(10,6)
    O5=Options('O5',i,t,goal)
    runQGame(env,option=O5)

    i=getCombs(range(8,12),range(7,12),(10,6))
    t=set(getCombs(range(14),range(14)))-set(getCombs(range(8,12),range(7,12)))
    goal=(7,9)
    O6=Options('O6',i,t,goal)
    runQGame(env,option=O6)

    i=getCombs(range(7,12),range(1,6),(10,6))
    t=set(getCombs(range(14),range(14)))-set(getCombs(range(7,12),range(1,6)))
    goal=(6,2)
    O7=Options('O7',i,t,goal)
    runQGame(env,option=O7)

    i=getCombs(range(7,12),range(1,6),(6,2))
    t=set(getCombs(range(14),range(14)))-set(getCombs(range(7,12),range(1,6)))
    goal=(10,6)
    O8=Options('O8',i,t,goal)
    runQGame(env,option=O8)

    return [O1,O2,O3,O4,O5,O6,O7,O8]

def runOptionsQGame(env,options,episodes=1000,alpha=0.1,epsilon=0.2,gamma=0.9,render=False,verbose=False,mainGoal=(7,9),showpolicy=False):
    optionsPerState=6 #2+4
    Q = np.zeros([env.observation_space.spaces[0].n,env.observation_space.spaces[1].n, optionsPerState]) #2 options 4 actions
    low_renderer=False
    AllSteps=[]
    AllRewards=[]
    for episode in range(episodes):
        done = False
        G=0
        state_t = env.reset(Gui=render,goal=mainGoal)#,InitiationSet=[(9,3)])
        steps=0
        epsilon *= 1# 0.996
        while not done:
            steps+=1
            if steps>1000:
                break
            #epsilon-greedy
            action = epsilonChoice(epsilon,Q[state_t[0],state_t[1]],optionsPerState) #2 - number of hallways per room
            
            if episode>(episodes-episodes/100):
                action = epsilonChoice(0,Q[state_t[0],state_t[1]],optionsPerState)
                
            if action<2:
                optionFlag=True 
                ops=getHall(state_t,options)[action]
                state_tp1, reward, done,k = playOption(env,state_t,options[ops],low_renderer)
                steps+=(k-1)
            else:
                optionFlag=False
                ops=action-2
                state_tp1, reward, done ,_ = env.step(ops)
                k=1
            if episode>(episodes-episodes/100) and render:
                low_renderer=True
                env.render()
                if verbose:
                    if optionFlag:
                        print 'O'+str(ops),
                    else:
                        print 'A'+str(ops),
                    if done:
                        print ''
                #print state_t,action,reward,state_tp1
            
            Q[state_t[0],state_t[1],action] += alpha * (reward + (gamma**k)*np.max(Q[state_tp1[0],state_tp1[1]]) - Q[state_t[0],state_t[1],action]) 
            G += reward
            state_t = state_tp1
            #print episode# % (episodes/10) == 0   
        AllSteps.append(steps)
        AllRewards.append(G)

        if episode % (episodes/10) == 0 and verbose==True:
            print 'Episode: {} Return : {} Steps: {}'.format(episode,G,steps)
    #print np.argmax(Q,axis=2)
    if showpolicy:
        showPolicy(Q,mainGoal)
    return AllSteps,AllRewards

def playIntraOption(env,state,option,render,Q,alpha,gamma,optionNum): # should be called only when agent is in initiation of the option
    option_reward=0
    done=False
    steps=1
    action=option.Policy[tuple(state)]
    obs,rew,done,_=env.step(action)
    option_reward+=rew
    #learning step
    updateFlag=False #to avoid double updates
    Q[state[0],state[1],action+2] = Q[state[0],state[1],action+2] + alpha *(rew + gamma*np.max(Q[tuple(obs)]) - Q[state[0],state[1],action+2])
    if tuple(state) not in option.Termination:
        Q[state[0],state[1],optionNum] = Q[state[0],state[1],optionNum] + alpha *(rew + gamma*Q[obs[0],obs[1],optionNum] - Q[state[0],state[1],optionNum])
    else:
        Q[state[0],state[1],optionNum] = Q[state[0],state[1],optionNum] + alpha *(rew + gamma*np.max(Q[tuple(obs)]) - Q[state[0],state[1],optionNum])
        updateFlag=True
    state =obs

    while not done and tuple(state) not in option.Termination:
        updateFlag=False
        obs,rew,done,_=env.step(option.Policy[tuple(state)])
        if render:
            env.render()
        #learning step
        Q[state[0],state[1],action+2] = Q[state[0],state[1],action+2] + alpha *(rew + gamma*np.max(Q[tuple(obs)]) - Q[state[0],state[1],action+2])
        Q[state[0],state[1],optionNum] = Q[state[0],state[1],optionNum] + alpha *(rew + gamma*Q[obs[0],obs[1],optionNum] - Q[state[0],state[1],optionNum])
        state=obs
        steps+=1
        option_reward=rew + option_reward
    if not updateFlag:
        Q[state[0],state[1],optionNum] = Q[state[0],state[1],optionNum] + alpha *(rew + gamma*np.max(Q[tuple(obs)]) - Q[state[0],state[1],optionNum])
    return state,option_reward,done,steps


def runIntraOptionsQGame(env,options,episodes=1000,alpha=0.1,epsilon=0.9,gamma=0.9,render=False,verbose=False,mainGoal=(7,9),showpolicy=False):
    optionsPerState=6 #2+4
    Q = np.zeros([env.observation_space.spaces[0].n,env.observation_space.spaces[1].n, optionsPerState]) #2 options 4 actions
    low_renderer=False
    AllSteps=[]
    AllRewards=[]
    for episode in range(episodes):
        done = False
        G=0
        state_t = env.reset(Gui=render,goal=mainGoal)#,InitiationSet=[(9,3)])
        steps=0
        #epsilon *= 0.996
        while not done:
            steps+=1
            if steps>1000:
                break
            #epsilon-greedy
            action = epsilonChoice(epsilon,Q[state_t[0],state_t[1]],optionsPerState) #2 - number of hallways per room
            
            if episode>(episodes-episodes/100):
                action = epsilonChoice(0,Q[state_t[0],state_t[1]],optionsPerState)
                
            if action<2:
                optionFlag=True 
                ops=getHall(state_t,options)[action]
                state_tp1, reward, done,k = playIntraOption(env,state_t,options[ops],low_renderer,Q,alpha,gamma,action)
                steps+=(k-1)
            else:
                optionFlag=False
                ops=action-2
                state_tp1, reward, done ,_ = env.step(ops)
                k=1
            if episode>(episodes-episodes/100) and render:
                low_renderer=True
                env.render()
                if verbose:
                    if optionFlag:
                        print 'O'+str(ops),
                    else:
                        print 'A'+str(ops),
                    if done:
                        print ''
                #print state_t,action,reward,state_tp1
            
            Q[state_t[0],state_t[1],action] += alpha * (reward + (gamma**k)*np.max(Q[state_tp1[0],state_tp1[1]]) - Q[state_t[0],state_t[1],action]) 
            G += reward
            state_t = state_tp1
            #print episode# % (episodes/10) == 0   
        AllSteps.append(steps)
        AllRewards.append(G)

        if episode % (episodes/10) == 0 and verbose==True:
            print 'Episode: {} Return : {} Steps: {}'.format(episode,G,steps)
    #print np.argmax(Q,axis=2)
    if showpolicy:
        showPolicy(Q,mainGoal)
    return AllSteps,AllRewards


