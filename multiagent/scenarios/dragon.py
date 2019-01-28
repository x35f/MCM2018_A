import numpy as np
from math import sqrt
from multiagent.core import World, Agent, Landmark,ENV_TYPE
from multiagent.scenario import BaseScenario

num_dragon=1
num_l_anim=0
num_s_anim=0
num_tree=0
num_home=0
num_agents=0
flight_horizon=0.1
ground_horizon=0.1


env_type=ENV_TYPE.Temperate
dragon_init_quality=40


class Scenario(BaseScenario):
    def set_env(self,env_type,world):
        if env_type==ENV_TYPE.Arctic:
            self.dens_l_anim=0.449
            self.dens_s_anim=1.0
            range_coef=0.0136
        elif env_type==ENV_TYPE.Arid:
            self.dens_l_anim=0.019
            self.dens_s_anim=0.3
            range_coef=0.3961
        elif env_type==ENV_TYPE.Temperate:
            self.dens_l_anim=0.862
            self.dens_s_anim=3.4
            range_coef=0.005
        elif env_type==ENV_TYPE.Model:
            self.dens_l_anim=0.457
            self.dens_s_anim=1.567
            range_coef=0.4
        else:
            raise NotImplementedError("Not implemented environment")
        init_home_range=range_coef*(dragon_init_quality**1.8)
        #print("init home range: ",init_home_range)
        self.num_l_anim=int(init_home_range*self.dens_l_anim)
        self.num_s_anim=int(init_home_range*self.dens_s_anim)
        #print("init l:",num_l_anim,"\ts:",num_s_anim)

    def make_world(self):
        world = World()
        world.set_env(env_type)
        self.set_env(env_type,world)

        # set any world properties first
        world.dim_c = 2
        #print("init l:",num_l_anim,"\ts:",num_s_anim)
        num_agents = num_dragon+self.num_l_anim+self.num_s_anim
        world.num_agents = num_agents

        num_adversaries = num_dragon
        num_landmarks = num_tree+num_home
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            if i<num_dragon:
                world.set_dragon(i)

            elif i<num_dragon+self.num_l_anim:
                world.set_l(i)
            elif i<num_dragon+self.num_l_anim+self.num_s_anim:
                world.set_s(i)
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            if i==num_landmarks-1:
                landmark.name="home"
                landmark.size = 0.06
            else:
                landmark.name = 'landmark %d' % i
                landmark.size = 0.02
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        print("reset world")
        for i, agent in enumerate(world.agents):
            if i<num_dragon:
                world.set_dragon(i)
            elif i<num_dragon+self.num_l_anim:
                world.set_l(i)
            elif i<num_dragon+self.num_l_anim+self.num_s_anim:
                world.set_s(i)
            agent.silent = True
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.0, 0.15, 0.0])
        # set goal landmark
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it

        return -dis(agent.base_pos-agent.state.p_pos)

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark"

        return agent.mass+agent.fat-dis(agent.base_pos-agent.state.p_pos)*agent.mass

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame

        entity_pos = []
        home_pos=[]
        for i,entity in enumerate(world.landmarks):
            if i<num_tree:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            else:
                home_pos=entity.state.p_pos - agent.state.p_pos
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        large_n=0
        small_n=0
        #larges=[]
        #smalls=[]
        for other in world.agents:
            if other is agent: continue
            #print(other.name," ",other.state.p_pos," <-> ",agent.state.p_pos)
            if dis(other.state.p_pos - agent.state.p_pos)<agent.horizon:
                if "large" in other.name and agent.quality>world.l_biomas*0.3:
                    large_n+=1
                    #larges.append(other.name)
                elif 'small' in other.name:
                    small_n+=1
                    #smalls.append(other.name)
        #if large_n+small_n>0:
            #print('obs',' ',larges,' ',smalls)
        return [large_n,small_n]

def dis(a):
    return sqrt(a[0]*a[0]+a[1]*a[1])
