import time
import numpy as np
import torch
from gym import spaces
from inspect import getargspec
from pprint import pprint
import imageio

try:
    import pyglet
except ImportError as e:
    reraise(
        suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *
except ImportError as e:
    reraise(prefix="Error occured while running `from pyglet.gl import *`",
            suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")








class GymWrapper(object):
    '''
    for multi-agent
    '''
    def __init__(self, env):
        self.env = env

    @property
    def observation_dim(self):
        '''
        for multi-agent, this is the obs per agent
        '''

        # tuple space
        if hasattr(self.env.observation_space, 'spaces'):
            total_obs_dim = 0
            for space in self.env.observation_space.spaces:
                if hasattr(self.env.action_space, 'shape'):
                    total_obs_dim += int(np.prod(space.shape))
                else: # Discrete
                    total_obs_dim += 1
            return total_obs_dim
        else:
            return int(np.prod(self.env.observation_space.shape))

    @property
    def num_actions(self):
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return int(self.env.action_space.nvec[0])
        elif hasattr(self.env.action_space, 'n'):
            # Discrete
            return self.env.action_space.n

    @property
    def dim_actions(self):
        # for multi-agent, this is the number of action per agent
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return self.env.action_space.shape[0]
            # return len(self.env.action_space.shape)
        elif hasattr(self.env.action_space, 'n'):
            # Discrete => only 1 action takes place at a time.
            return 1

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, epoch):
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            obs = self.env.reset(epoch)
        else:
            obs = self.env.reset()

        obs = self._flatten_obs(obs)
        return obs

    def display(self):
        self.env.render()
        time.sleep(0.02)

    def end_display(self):
        self.env.exit_render()

    def step(self, action):
        # TODO: Modify all environments to take list of action
        # instead of doing this
        if self.dim_actions == 1:
            action = action[0]
        obs, r, done, info = self.env.step(action)
        obs = self._flatten_obs(obs)
        return (obs, r, done, info)

    def reward_terminal(self):
        if hasattr(self.env, 'reward_terminal'):
            return self.env.reward_terminal()
        else:
            return np.zeros(1)

    def _flatten_obs(self, obs):
        if isinstance(obs, tuple):
            _obs=[]
            for agent in obs: #list/tuple of observations.
                ag_obs = []
                for obs_kind in agent:
                    ag_obs.append(np.array(obs_kind).flatten())
                _obs.append(np.concatenate(ag_obs))
            obs = np.stack(_obs)

        obs = obs.reshape(1, -1, self.observation_dim)
        obs = torch.from_numpy(obs).double()
        return obs

    def get_stat(self):
        if hasattr(self.env, 'stat'):
            self.env.stat.pop('steps_taken', None)
            return self.env.stat
        else:
            return dict()


class EnvWrapper(object):
    '''
    for multi-agent
    '''

    def __init__(self, env):
        self.env = env
        self.video = []
        self.video_name = ''

    @property
    def observation_dim(self):
        '''
        for multi-agent, this is the obs per agent
        '''
        # tuple space
        return np.prod(self.env.observation_space[0].shape)

    @property
    def num_actions(self):
        return self.action_space[0].n
        

    @property
    def dim_actions(self):
        # for multi-agent, this is the number of action per agent
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return self.env.action_space.shape[0]
            # return len(self.env.action_space.shape)
        elif hasattr(self.env.action_space, 'n'):
            # Discrete => only 1 action takes place at a time.
            return 1
        else :
            return 1

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, epoch):
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            obs = self.env.reset(epoch)
        else:
            obs = self.env.reset()

        obs = self._flatten_obs(obs)
        return obs

    def start_record(self,name):
        self.video_name = name
        pyglet.image.get_buffer_manager().get_color_buffer().get_image_data().save(name+'.png')
        self.video.append(imageio.imread(name+'.png'))


    def end_record(self):
        imageio.mimsave(self.video_name+'.gif', self.video, 'GIF')
        self.video_name = ''
        self.video = []


    def display(self):
        self.env.render()

        


    def end_display(self):
        self.env._reset_render()

    def step(self, action):
        # TODO: Modify all environments to take list of action
        # instead of doing this
        if self.dim_actions == 1:
            action = action[0]
        obs, r, done, info = self.env.step(action)
        r = np.array(r)
        obs = self._flatten_obs(obs)

        return (obs, r, done, info)

    def reward_terminal(self):
        if hasattr(self.env, 'reward_terminal'):
            return self.env.reward_terminal()
        else:
            return np.zeros(1)

    def _flatten_obs(self, obs):
        max_dim = 0
        for _ in obs:
            max_dim = max(max_dim, len(_))
        for x in range(len(obs)):
            if len(obs[x]) < max_dim:
                for i in range(max_dim-len(obs[x])):
                    obs[x] = np.append(obs[x], 0)
        obs = np.array(obs)
        obs = obs.reshape(1, -1, self.observation_dim)
        obs = torch.from_numpy(obs).double()
        return obs

    def get_stat(self):
        if hasattr(self.env, 'stat'):
            self.env.stat.pop('steps_taken', None)
            return self.env.stat
        else:
            return dict()
