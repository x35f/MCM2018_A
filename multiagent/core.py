import numpy as np
from math import sqrt
from enum import Enum
def dis(a,b):
    dx=a[0]-b[0]
    dy=a[1]-b[1]
    return sqrt(dx*dx+dy*dy)

dragon_init_quality=40

dragon_init_pos=np.array([0.5,0.5])
flight_horizon=0.1
ground_horizon=0.1
#agent self prob
l_biomas=0.0
s_biomas=0.0
#eaten prob
l_eaten_prob=0.75
s_eaten_prob=0.75
fire_prob=0.5
fire_cost=0.0
#home_range
drag_range=1
l_range=0.2
s_range=0.001
tree_range=0.0

#dragon property
fat_perc=0.25
hunt_energy_cost=0.05
regular_energy_cost=0.01 #per 100 timestep
convert_perc=0.1
convert_fat_perc=0.5

exp_income=0.0045*dragon_init_quality**0.75+1.2*dragon_init_quality**0.22+0.1*dragon_init_quality
day_step=100
max_mass=540
# physical/external base state of all entites

class ENV_TYPE(Enum):
    Arctic=1
    Arid=2
    Temperate=3
    Model=4

class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        self.fat=0.0
        self.quality=0.0
    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        #hipoint
        self.base_pos=[0.0,0.0]
        self.alive=True
        self.eaten_prob=0.0
        #dragon property
        self.energy=100
        self.velocity=0
        self.hunt_energy_cost=0
        self.convert_perc=convert_perc
        self.convert_fat_perc=convert_fat_perc
        self.regular_energy_cost=regular_energy_cost
        self.exp_income=exp_income
        #pray property
        self.grouth_prob=0.01 # by group size of  hundred in 1 timestep
        self.horizon=0.01
# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.goal=[]
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks
    @property
    def dragon(self):
        return self.agents[0]
    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def set_env(self,env_type):
        if env_type==ENV_TYPE.Arctic:
            self.dens_l_anim=0.49
            self.dens_s_anim=1.0
            self.l_abs_radius=14.6
            self.s_abs_radius=2.82
            self.l_biomas=406
            self.s_biomas=7.85
            self.home_range_coef=0.0136
            self.l_growth_coef=0.008
            self.s_growth_coef=0.032
            self.hunt_success_prob=0.625
        elif env_type==ENV_TYPE.Arid:
            self.dens_l_anim=0.019
            self.dens_s_anim=0.3
            self.l_abs_radius=12.4
            self.s_abs_radius=3
            self.l_biomas=52
            self.s_biomas=10.6
            self.home_range_coef=0.3961
            self.l_growth_coef=0.011
            self.s_growth_coef=0.016
            self.hunt_success_prob=0.875
        elif env_type==ENV_TYPE.Temperate:
            self.dens_l_anim=0.862
            self.dens_s_anim=3.4
            self.l_abs_radius=31.4
            self.s_abs_radius=14.93
            self.l_biomas=340
            self.s_biomas=27
            self.home_range_coef=0.005
            self.l_growth_coef=0.0054
            self.s_growth_coef=0.022
            self.hunt_success_prob=0.75
        elif env_type==ENV_TYPE.Model:
            self.dens_l_anim=0.457
            self.dens_s_anim=1.567
            self.l_abs_radius=12.43
            self.s_abs_radius=2.82
            self.l_biomas=360
            self.s_biomas=22
            self.home_range_coef=0.4
            self.l_growth_coef=0.0082
            self.s_growth_coef=0.0219
            self.hunt_success_prob=0.75
        else:
            raise NotImplementedError("Not implemented environment")
        #self.set_dragon_home_range(dragon_init_quality)

    def set_dragon_home_range(self):
        m=self.dragon.quality
        self.agents[0].home_range=self.home_range_coef*(m**1.8)
        #print("m:" ,m , 'range:',self.agents[0].home_range)
        self.agents[0].home_radius=sqrt(self.dragon.home_range/3.14)
        #print("init range:", self.dragon_home_range,"\t radius:",self.dragon_home_radius)

    def relative_home_range(self,entity):
        if 'large' in entity.name:
            return self.l_abs_radius/self.dragon.home_radius
        elif 'small' in entity.name:
            return self.s_abs_radius/self.dragon.home_radius
        elif 'dragon' in entity.name:
            return 1.0
        else:
            raise NotImplementedError("Not implemented environment")

    def step(self):
        # set actions for scripted agents

        #agent=self.agents[0]
        #print("num agents: ",len(self.agents),' ',agent.name)
        #agent.action = agent.action_callback(agent, self)

        # gather forces applied to entities
        p_force = [None] * len([0])
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        #p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        self.update_agent_state(self.agents[0])

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate([self.agents[0]]):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate([self.entities[0]]):
            if not entity.movable: continue
            #entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel = (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt
            #restrain to a range\
            if isinstance(entity,Agent):
                self.limit_range(i)

    def limit_range(self,agent_id):
        entity=self.entities[agent_id]
        [x,y]=entity.state.p_pos

        dist=dis([x,y],entity.base_pos)
        if dist>self.relative_home_range(entity):
            dx=x-entity.base_pos[0]
            dy=y-entity.base_pos[1]
            dx/=dist
            dy/=dist
            x=entity.base_pos[0]+dx
            y=entity.base_pos[1]+dy
        x=max(-1.0,x)
        x=min(1.0,x)
        y=max(-1.0,y)
        y=min(1.0,y)
        entity.state.p_pos=np.array([x,y])


    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise
        #eaten



    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    def set_dragon(self,id=0):
        self.agents[id].name = 'dragon %d' % id
        self.agents[id].collide = False
        self.agents[id].adversary = True
        self.agents[id].size = 0.02
        self.agents[id].horizon=flight_horizon
        self.agents[id].home_range=drag_range
        self.agents[id].hunt_energy_cost=hunt_energy_cost
        self.agents[id].state.p_pos=np.array([0.0,0.0])
        self.agents[id].state.p_vel=np.zeros(self.dim_p)
        self.agents[id].base_pos=np.array([0.0,0.0])
        self.agents[id].state.c = np.zeros(self.dim_c)
        self.agents[id].color = np.array([0.35, 0.0, 0.0])
        self.agents[id].silent=True
        self.agents[id].quality=dragon_init_quality
        self.agents[id].fat=dragon_init_quality*fat_perc
        self.agents[id].stationary_cost=0.0045*(dragon_init_quality**0.75)
        self.agents[id].patrol_cost=1.2*(dragon_init_quality**0.22)
        self.agents[id].fire_cost=hunt_energy_cost
        self.agents[id].exp_income=exp_income
        self.agents[id].daily_income=exp_income
        self.agents[id].max_mass=max_mass
        self.set_dragon_home_range()
    def set_l(self,id=0):
        self.agents[id].name = 'large %d' % id
        self.agents[id].collide = False
        self.agents[id].adversary = False
        self.agents[id].size = 0.01
        self.agents[id].biomas=self.l_biomas
        self.agents[id].eaten_prob=l_eaten_prob
        self.agents[id].home_range=l_range
        pos=np.random.uniform(-1, +1, self.dim_p)
        self.agents[id].state.p_pos=pos
        self.agents[id].state.p_vel=np.zeros(self.dim_p)
        self.agents[id].base_pos=pos
        self.agents[id].state.c = np.zeros(self.dim_c)
        self.agents[id].color = np.array([0.0, 0.0, 0.35])
        self.agents[id].silent=True
        #self.agents[id].fire_eaten_prob=l_fire_eaten_prob
    def set_s(self,id=0):
        self.agents[id].name= 'small %d' %id
        self.agents[id].collide=False
        self.agents[id].adversary=False
        self.agents[id].size=0.005
        self.agents[id].biomas=self.s_biomas
        self.agents[id].eaten_prob=s_eaten_prob
        self.agents[id].home_range=s_range
        pos=np.random.uniform(-1, +1, self.dim_p)
        self.agents[id].state.p_pos=pos
        self.agents[id].state.p_vel=np.zeros(self.dim_p)
        self.agents[id].base_pos=pos
        self.agents[id].state.c = np.zeros(self.dim_c)
        self.agents[id].color=np.array([0.2,0.2,0.2])
        self.agents[id].silent=True
        #self.agents[id].fire_eaten_prob=s_fire_eaten_prob
