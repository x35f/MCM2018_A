from make_env import make_env
from random import randint
from random import random
import time
from math import tan
from enum import Enum
from numpy.random import choice
import argparse
import pickle
import os
from os import path
base_path=os.getcwd()
data_path=path.abspath(path.join(base_path,'data'))
print("data_path")
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env', default='Temperate')
parser.add_argument('--s-density',type=float,default=0.0)
parser.add_argument('--l-density',type=float,default=0.0)
parser.add_argument('--max-home-range',type=int,default=0.0)
parser.add_argument('--range-coef',type=float,default=0.0)
parser.add_argument('--mass',type=float,default=40.0)
#parser.add_argument('--none',dafault=0)
args = parser.parse_args()

day_step=100
digest_time=450
env=make_env('dragon',args)
obs=env.reset()
num_agents=env.n
step_length=10
#env.render()
dead=False
info=[]
#print(env.action_space)
action_space=[0,1,2,3,4,5]
def normalize(list):
    s=sum(list)
    if s>1e-6:
        list=[list[i]/s for i in range(len(list))]
    return list
def gene_action_prob(degree):
    #stop right left up down
    rad=(degree%180)/180*3.14
    prob=[0.0 for i in range(6)]
    if degree==0:
        prob[2]=1.0
    elif 0<=degree<90:
        val=tan(rad)
        prob=[0.0,1.0,0.0,val,0.0,0.0]
    elif degree==90:
        prob[3]=1.0
    elif 90<degree<180:
        val=-tan(rad)
        prob=[0.0,0.0,1.0,val,0.0,0.0]
    elif degree==180:
        prob[2]=1.0
    elif 180<degree<270:
        val=tan(rad)
        prob=[0.0, 0.0 ,1.0, 0.0, val,0.0]
    elif degree==270:
        prob[4]=1.0
    elif 270<degree<360:
        val=-tan(rad)
        prob=[0.0,1.0,0.0,0.0,val,0.0]
    #print(degree," ",prob)
    return normalize(prob)
def choice(prob):
    seed=random()
    bin_count=0.0
    for i in range(6):
        bin_count+=prob[i]
        if seed<=bin_count+1e-5:
            return i

digest_count=0
direction=randint(0,359)
intake=100.0
forward_count=step_length
action_prob=gene_action_prob(direction)
prev_action_prob=[]
eps=0
maximum_recorded_step=365
# memory structure( mass, fat, intake, large, small ,home range)
sample_memory=[0.0 for i in range(10)]
print("Environment:",args.env)
print("Day\tmass\tfat\tintake\tlarge\tsmall\thome_range")
while not dead:
    eps+=1
    action=[0 for i in range(6)]
    if digest_count>0:
        digest_count-=1
        action[0]=1
    else:
        if intake>0:
            #print(obs)
            if sum(obs)>0:#food in range
                #print(obs)
                action[5]=1
            elif forward_count>0:#forward
                forward_count-=1
                #print("forwarding")
                action[choice(action_prob)]=1
            else:
                direction=randint(0,359)
                forward_count=step_length
                action_prob=gene_action_prob(direction)
                action[choice(action_prob)]=1
        else:

            digest_count=digest_time
            action[0]=1
            #print("\033[31mdigesting\033[0m")
    #print(action," ",action_prob," ",forward_count," ",digest_count," ",intake)
    obs,rew,done,info=env.step(action)
    intake=info['intake']
    if eps%day_step==0:
        print("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\t{:.2f}".format(
            int(eps/day_step),info['mass'],info['fat'],info['intake'],info['large'],info['small'],info['home_range']))
        digest_time=2*(info['mass']**0.22)*day_step
        #time.sleep(0.5)
    if info['fat']<0:
        print("Your fucking dragon is dead")
        break
    if eps/day_step>=366:
        break
    #env.render()
