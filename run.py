import sys
import numpy as np
import gym
import Env
import itertools as it
from options import runOptionsQGame,GetOptions,playOption,runIntraOptionsQGame
from helper import save_data,load_data
import argparse

def checkArgs(args=None):
    parser = argparse.ArgumentParser(description='Grid World')
    parser.add_argument('--alpha',help = 'Learning Rate(alpha)', default = 0.001,type=float)
    parser.add_argument('--epsilon',help = 'epsilon for exploration', default= 0.2,type=float)
    parser.add_argument('--gamma',help = 'Discount rate', default=0.9,type=float)
    parser.add_argument('--episodes',help = 'Number of episodes',default=20000,type=int)
    parser.add_argument('--verbose',help = 'Verbosity', action='store_true',default=False)
    parser.add_argument('--goal',help = 'G1/G1', default='G1')
    parser.add_argument('--render',help = 'Render Environment', action='store_true',default=False)
    parser.add_argument('--show-policy',help='Show policy at end',action='store_true',default=False)
    parser.add_argument('--algo',help='O/IO (options/intra-options)',default='O')
    parser.add_argument('--fo',help = 'Learn Fresh options', action='store_true',default=False)
    args = parser.parse_args(args)
    return args

args=checkArgs(sys.argv[1:])

print 'Starting with',args

env=gym.make('grid-v0')

if args.goal=='G1':
    goal=(7,9)
elif args.goal=='G2':
    goal=(9,9)
else:
    print 'invalid goal'
    sys.exit()


#save_data(GetOptions(env))

if args.fo:
    OPTIONS=GetOptions(env) #learning fresh options using Q 
else:
    OPTIONS=load_data() #loading saved options

if args.algo=='O':
    steps,rewards=runOptionsQGame(env,OPTIONS,verbose=args.verbose,render=args.render,mainGoal=goal,
    alpha=args.alpha,epsilon=args.epsilon, episodes=args.episodes,showpolicy=args.show_policy)
elif args.algo=='IO':
    steps,rewards=runIntraOptionsQGame(env,OPTIONS,verbose=args.verbose,render=args.render,mainGoal=goal,
    alpha=args.alpha,epsilon=args.epsilon, episodes=args.episodes,showpolicy=args.show_policy)
else:
    print 'invalid algo'
