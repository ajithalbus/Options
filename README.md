# Options
## Implementation of SMDP Q-Learning Options framework.

The enviroment is similar to the one specified in the original paper.

Requirements : pygame,gym,numpy,argparse
	
Usage:
	
	python2 run.py run.py [-h] [--alpha ALPHA] [--epsilon EPSILON] [--gamma GAMMA]
              [--episodes EPISODES] [--verbose] [--goal GOAL] [--render]
              [--show-policy] [--algo ALGO] [--fo]

	ALPHA : learning rate [Default:0.001]
	EPSILON : Epsilon value for epsilon-greedy exploration [Default:0.2]
	GAMMA : Discount [Default:0.9]
	--render : to render the game for last 10% of the episodes
	--show-policy : to show policy at the end
	--verbose : to print steps and rewards 
	ALGO : O/IO (Options/Intra-options) [Default:O]
	GOAL : G1/G2 (goal-1/goal-2) [Default:G1]
	--fo : learn fresh options #not recommended

	
Example :To run a game with default settings on goal-1 and show learnt policy at the end
	
	python run.py --verbose --show-policy
		
Reference : [Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)

P.S : rewards have been scaled to 10
