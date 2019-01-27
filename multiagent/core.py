import numpy as np
from math import sqrt
def dis(a,b):
    dx=a[0]-b[0]
    dy=a[1]-b[1]
    return sqrt(dx*dx+dy*dy)

num_dragon=1
num_l_anim=100
num_s_anim=1000
num_tree=0
num_home=1
num_agents=0
dragon_init_pos=np.array([0.5,0.5])
flight_horizon=0.1
ground_horizon=0.1
#agent self prob
l_biomas=0.5
s_biomas=0.2
#eaten prob
l_eaten_prob=0.8
s_eaten_prob=0.5
#home_range
drag_range=1
l_range=0.2
s_range=0.001
tree_range=0.0

#dragon property
fat_percentage=0.25
hunt_energy_cost=0.05
regular_energy_cost=0.01 #per 100 timestep
convert_perc=0.1
convert_fat_perc=0.25

# physical/external base state of all entites
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
        self.fat=fat_percentage
        self.quality=1.0
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
        self.exp_income=0
        self.hunt_range=0.0
        self.hunt_cost=100
        self.energy=100
        self.velocity=0
        self.hunt_energy_cost=0
        self.convert_perc=convert_perc
        self.convert_fat_perc=convert_fat_perc
        self.regular_energy_cost=regular_energy_cost
        #pray property
        self.grouth_prob=0.01 # by group size of  hundred in 1 timestep

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

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
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
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
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
        if dist>entity.home_range:
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
        self.agents[id].size = 0.05
        self.agents[id].hunt_range=flight_horizon
        self.agents[id].home_range=drag_range
        self.agents[id].hunt_energy_cost=hunt_energy_cost
        self.agents[id].state.p_pos=np.array([0.0,0.0])
        self.agents[id].state.p_vel=np.zeros(self.dim_p)
        self.agents[id].base_pos=np.array([0.0,0.0])
        self.agents[id].state.c = np.zeros(self.dim_c)
        self.agents[id].base_pos=self.agents[id].state.p_pos
        self.agents[id].color = np.array([0.35, 0.0, 0.0])
        self.agents[id].silent=True

    def set_l(self,id=0):
        self.agents[id].name = 'large %d' % id
        self.agents[id].collide = False
        self.agents[id].adversary = False
        self.agents[id].size = 0.03
        self.agents[id].biomas=l_biomas
        self.agents[id].eaten_prob=l_eaten_prob
        self.agents[id].home_range=l_range
        pos=np.random.uniform(-1, +1, self.dim_p)
        self.agents[id].state.p_pos=pos
        self.agents[id].state.p_vel=np.zeros(self.dim_p)
        self.agents[id].base_pos=pos
        self.agents[id].state.c = np.zeros(self.dim_c)
        self.agents[id].base_pos=self.agents[id].state.p_pos
        self.agents[id].color = np.array([0.0, 0.0, 0.35])
        self.agents[id].silent=True

    def set_s(self,id=0):
        self.agents[id].name= 'small %d' %id
        self.agents[id].collide=False
        self.agents[id].adversary=False
        self.agents[id].size=0.01
        self.agents[id].biomas=s_biomas
        self.agents[id].eaten_prob=s_eaten_prob
        self.agents[id].home_range=s_range
        pos=np.random.uniform(-1, +1, self.dim_p)
        self.agents[id].state.p_pos=pos
        self.agents[id].state.p_vel=np.zeros(self.dim_p)
        self.agents[id].base_pos=pos
        self.agents[id].state.c = np.zeros(self.dim_c)
        self.agents[id].base_pos=self.agents[id].state.p_pos
        self.agents[id].color=np.array([0.2,0.2,0.2])
        self.agents[id].silent=True
