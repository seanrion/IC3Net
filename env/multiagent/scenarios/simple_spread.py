import numpy as np
from env.multiagent.core import World, Agent, Landmark
from env.multiagent.scenario import BaseScenario



class Scenario(BaseScenario):
    def __init__(self, args):
        #智能体数量
        self.num_agents = args.nagents
        #地标数量
        self.num_landmarks = args.num_landmarks
        #距离门槛，即智能体离地标有多近算成功
        #舞台大小，关系到随机出生地距离舞台中心有多远
        self.arena_size = args.arena_size
        #智能体间是否合作
        self.collaborative = args.collaborative
        self.silent = args.silent
        self.agent_size = args.agent_size
        self.landmark_size = args.landmark_size


    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = self.num_agents
        num_landmarks = self.num_landmarks
        world.collaborative = self.collaborative
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = self.silent
            agent.size = self.agent_size
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = self.landmark_size

        # make initial conditions
        self.reset_world(world)
        return world

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if self.is_collision(l, agent):
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

        
    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-self.arena_size, self.arena_size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-self.arena_size, self.arena_size, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # rew = 0
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
        #              for a in world.agents]
        #     rew -= min(dists)
        # if agent.collide:
        #     for a in world.agents:
        #         if a is agent:
        #             continue
        #         if self.is_collision(a, agent):
        #             rew -= 1
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            rew -= min(dists)

        for a in world.agents:
            if a is agent:
                continue
            if self.is_collision(a, agent):
                dist_value = 1
            else:
                dist_value = np.exp(-np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))))
            rew -= dist_value
        # vels = [np.sqrt(np.sum(np.square(a.state.p_vel)))for a in world.agents]
        # vel = min(vels)
        # rew += vel if vel<1 else 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)


    def done(self, agent, world):
        index = 0
        is_collision = []
        for i, a in enumerate(world.agents):
            if a is agent:
                index = i
        for a in world.agents:
            is_collision.append(self.is_collision(a, world.landmarks[index]))
        
        return is_collision.count(True)==1



    def info(self, agent, world):
        return []
    
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
