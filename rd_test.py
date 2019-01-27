from make_env import make_env
from random import randint
import time
day_steps=100

env=make_env('dragon')
obs=env.reset()
num_agents=env.n
num_eps=100
max_act_num=100
env.render()
#print(env.action_space)
act_dim=[6]
for i in range(1000):
    act_dim.append(5)

def gene_action(agent_id):
    action=[0 for i in range(act_dim[agent_id])]
    action[randint(0,act_dim[agent_id]-1)]=1
    return action

for eps in range(1000000000):
    action=gene_action(0)
    #print(action_s)
    obs,rew,done_s,info=env.step(action)
    #env.render()
    if (eps%10==0):
        print("Day ",eps/10,"\tMass: ",info['mass'],"Fat: ",info['fat'])
    #print(eps)
