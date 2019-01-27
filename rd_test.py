from make_env import make_env
from random import randint
import time
day_steps=100

env=make_env('dragon')
obs=env.reset()
num_agents=env.n
num_eps=100
max_act_num=100
obs=[0,0]
info=[]
#print(env.action_space)
def gene_action(id):
    act=[0 for i in range(6)]
    act[randint(0,5)]=1
    return act
for eps in range(1000000000):
    action=gene_action(0)
    #print(action_s)
    if sum(obs)==0 or info['daily_income']<=0:
        action[5]=0
    obs,rew,done,info=env.step(action)
    #env.render()
    if (eps%100==0):
        print("Day ",eps/100,"\tMass: ",info['mass'],"\tFat: ",info['fat'],\
            "\texp_income:",info['daily_income'],"\tl+s",info['large_n'],"+",info['small_n'])
        #print("range:",info['home_range'])
    #print(eps)
