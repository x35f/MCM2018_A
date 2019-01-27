from make_env import make_env
from random import randint
import time
day_steps=100

env=make_env('dragon')
obs=env.reset()
num_agents=env.n
num_eps=100
max_act_num=100
#print(env.action_space)
def gene_action(id):
    act=[0 for i in range(6)]
    act[randint(0,5)]=1
    return act
for eps in range(1000000000):
    action=gene_action(0)
    #print(action_s)
    obs,rew,done,info=env.step(action)
    env.render()
    if (eps%10==0):
        print("Day ",eps/10,"\tMass: ",info['mass'],"\tFat: ",info['fat'],\
            "\tl+s",info['large_n'],"+",info['small_n'])
    #print(eps)
