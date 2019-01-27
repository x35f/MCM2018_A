import gym
from math import sqrt
from gym import spaces
from gym.envs.registration import EnvSpec
from multiagent.core import Agent
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
from random import random,randint
import random
from enum import Enum
# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
def dis(a,b):
    dx=a[0]-b[0]
    dy=a[1]-b[1]
    return sqrt(dx*dx+dy*dy)

s_birth_prob=0.01 #by group of a hundred
l_birth_prob=0.001 #by group of a hundred

day_step=100

class ENV_TYPE(Enum):
    Arctic=1
    Arid=2
    Temperate=3

env_type=ENV_TYPE.Arctic

class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        self.step_count=1
        self.world = world
        self.agents = self.world.policy_agents
        self.num_l=0
        self.num_s=0
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        self.n_dragon=0
        self.n_l=0
        self.n_s=0
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            if "dragon" in agent.name:
                self.n_dragon+=1
            elif "large" in agent.name:
                self.n_l+=1
            else:
                self.n_s+=1
            total_action_space = []
            # physical action space

            if self.discrete_action_space:
                if agent.adversary:
                    u_action_space = spaces.Discrete(world.dim_p * 2 + 1+1)
                else:
                    u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self.changed_in_step=True
        self.act_space=self.action_space[0]
        self.obs_space=self.observation_space[0]
        self._reset_render()

    @property
    def dragon(self):
        return self.agents[0]
    def gene_rand_actions():
        add_actions=[]

    def record_dragon_init_state(self):
        self.dragon_init_home_range=self.dragon.home_range
        self.d_mass=0.0
        self.d_fat=0.0
    def gene_random_actions(self):
        actions=[]
        for i in range(self.n):
            action=[0 for i in range(5)]
            action[randint(0,4)]=1
            actions.append(action)
        return actions

    def step(self, action):
        self.step_count+=1
        self.record_dragon_init_state()
        if action[0]==1:
            self.d_fat-=self.dragon.stationary_cost/day_step
        else:
            self.d_fat-=self.dragon.patrol_cost/day_step
        action_n=[action]

        action_n=action_n+self.gene_random_actions()

        info = {'n': []}
        self.changed_in_step=False
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            #print(len(self.agents)," ",len(action_n)," ",len(self.action_space))
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        self.changed_in_step=False
        if self.agents[0].adversary:
            if action_n[0][5]==1:
                self.hunt_step()
        # record observation for each agent

        # all agents get total reward in cooperative case
        obs=self._get_obs(self.dragon)
        rew=self._get_reward(self.dragon)
        done=self._get_done(self.dragon)
        self.growth_step()
        self.world_update()
        info['mass']=self.dragon.quality
        info['fat']=self.dragon.fat
        info['large_n']=self.n_l
        info['small_n']=self.n_s
        info['daily_income']=self.dragon.daily_income
        info['home_range']=self.dragon.home_range
        return obs, rew, done, info

    def new_l_agent(self):
        self.world.agents.append(Agent())
        self.world.set_s(len(self.world.agents)-1)
        #print(tn," -> ",len(self.w))
        self.agents.append(self.world.agents[len(self.world.agents)-1])
        self.action_space.append(self.new_act_space())
    def new_s_agent(self):
        self.world.agents.append(Agent())
        self.world.set_s(len(self.world.agents)-1)
        self.agents.append(self.world.agents[len(self.world.agents)-1])
        self.action_space.append(self.new_act_space())

    def growth_step(self):
        s_seed=random.random()
        if s_seed<s_birth_prob:
            self.new_s_agent()
            self.changed_in_step=True

        l_seed=random.random()
        if l_seed<l_birth_prob:
            self.new_l_agent()
            self.changed_in_step=True

    def world_update(self):

        """if self.changed_in_step:
            print(len(self.agents)," ",len(self.world.agents))"""
        """update dragon property"""
        self.dragon_state_update()
        self.environment_update()

    def dragon_state_update(self):
        #self.dragon.mass

        self.dragon.daily_income-=self.d_mass
        d_mass=self.d_mass*self.dragon.convert_perc
        self.dragon.fat+=self.d_fat
        #print("mass:",self.dragon.mass)
        self.dragon.quality+=d_mass
        self.dragon.fat+=d_mass*self.dragon.convert_fat_perc
        m=self.dragon.quality
        self.dragon.hunt_cost=0.05*m
        self.dragon.patrol_cost=1.2*(m**0.22)
        self.dragon.stationary_cost=0.0045*(m**0.75)
        self.dragon.hunt_energy_cost=0.05*self.dragon.fat
        self.dragon.fire_cost=self.dragon.hunt_energy_cost
        self.dragon.exp_income=0.0045*(m**0.75)+1.2*(m**0.22)+0.05*m
        self.dragon.horizon=1.07*(m**0.68)


    def environment_update(self):
        #reset some parameters after a day
        if self.step_count%day_step ==0:
            #daily update for new_range income
            #print("reset to ",self.dragon.exp_income)
            self.dragon.daily_income=self.dragon.exp_income
            self.step_count=1
            prev_home_range=self.dragon.home_range
            self.world.set_dragon_home_range()
            d_range=self.dragon.home_range-prev_home_range
            d_l_num=int(d_range*self.world.dens_l_anim)
            d_s_num=int(d_range*self.world.dens_s_anim)
            #print("New in sight:",d_l_num," ",d_s_num)
            for i in range(d_l_num):
                self.new_l_agent()
            for i in range(d_s_num):
                self.new_s_agent()
            #daily update for new born animals
            growth_seed=random.random()
            ds=self.world.dens_l_anim*self.dragon.home_range
            n=self.n_l
            prob=self.world.l_growth_coef*(ds-n)/(ds*day_step)
            if growth_seed<prob:
                print("born l agent")
                self.new_l_agent()
            growth_seed=random.random()
            dl=self.world.dens_s_anim*self.dragon.home_range
            n=self.n_s
            prob=self.world.s_growth_coef*(ds-n)/(ds*day_step)
            if growth_seed<prob:
                print("born s agent")
                self.new_s_agent()

        self.world.entites=[]
        self.n_dragon=0
        self.n_l=0
        self.n_s=0
        for agent in self.agents:

            if "dragon" in agent.name:
                self.n_dragon+=1
            elif "large" in agent.name:
                self.n_l+=1
            else:
                self.n_s+=1
            self.world.entites.append(agent)
        #print("n_count:",self.n_l," ",self.n_s)
        for landmark in self.world.landmarks:
            self.world.entities.append(landmark)
        self.world.num_agents=len(self.agents)
        self.n=self.world.num_agents

    def hunt_step(self):
        #print("hunting with range",self.dragon.horizon)
        self.d_fat-=self.agents[0].hunt_energy_cost
        hunt_pos=self.agents[0].state.p_pos
        closest_l_agent_id=-1
        closest_s_agent_id=-1
        closest_l_dis=self.dragon.horizon
        closest_s_dis=self.dragon.horizon
        for i,agent in enumerate(self.agents):
            if i>=1:
                dist=dis(hunt_pos,agent.state.p_pos)
                if "large" in agent.name and self.dragon.quality>self.world.l_biomas:
                    if closest_l_dis>dist:
                        closest_l_agent_id=i
                        closest_l_dis=dist
                if "small" in agent.name:
                    if closest_s_dis>dist:
                        closest_s_agent_id=i
                        closest_s_dis=dist
        if closest_l_agent_id!=-1:
            self.eat_agent(closest_l_agent_id)
        elif closest_s_agent_id!=-1:
            self.eat_agent(closest_l_agent_id)
        else:
            raise NotImplementedError("No agent to hunt")


    def eat_agent(self,agent_id):
        seed=random.random()
        #print("prob:",self.agents[agent_id].eaten_prob)
        fire=random.choice([False,True])
        if fire:
            self.d_fat-=self.dragon.fire_cost
            if seed<self.agents[agent_id].eaten_prob:
                #print("eat large agent",self.agents[agent_id].name)
                self.d_mass+=self.world.l_biomas
                del self.agents[agent_id]
                del self.world.agents[agent_id]
                del self.action_space[agent_id]
                self.changed_in_step=True
                #print(len(self.agents))
        else:
            if seed<self.agents[agent_id].fire_eaten_prob:
                #print("biomas=",self.agents[agent_id].biomas)
                #print("eat small agent",self.agents[agent_id].name)
                self.d_mass+=self.world.s_biomas
                del self.agents[agent_id]
                del self.world.agents[agent_id]
                del self.action_space[agent_id]
                self.changed_in_step=True
                #print(len(self.agents))



    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        base_pos=[agent.base_pos for agent in self.agents]
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback==None:
            return None
        return self.info_callback(agent, self.world)


    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    def new_act_space(self):
        total_action_space = []
        u_act_space=spaces.Discrete(self.world.dim_p*2+1)
        total_action_space.append(u_act_space)
        return MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]
        #print("transferred action",action)
        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            #print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(1200,1200)

        # create rendering geometry
        if True:
            #recreate anyway
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space[0]

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space[0]

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done


        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
