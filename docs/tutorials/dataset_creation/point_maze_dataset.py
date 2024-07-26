"""
PointMaze D4RL dataset
=========================================
"""
# %%%
# In this tutorial you will learn how to re-create the Maze2D datasets from `D4RL <https://sites.google.com/view/d4rl/home>`_ [1] with Minari.
# We will be using the refactored version of the PointMaze environments in `Gymnasium-Robotics <https://robotics.farama.org/envs/maze/point_maze/>`_ which support the Gymnasium API as well as the latest
# MuJoCo python bindings.
#
# Lets start by breaking down the steps to generate these datasets:
#   1. First we need to create a planner that outputs a trajectory of waypoints that the agent can follow to reach the goal from its initial location in the maze. We will be using
#      `Q-Value Iteration <https://towardsdatascience.com/fundamental-iterative-methods-of-reinforcement-learning-df8ff078652a>`_ [2] to solve the discrete grid maze, same as in D4RL.
#   2. Then we also need to generate the actions so that the agent can follow the waypoints of the trajectory. For this purpose D4RL implements a PD controller.
#   3. Finally, to create the Minari dataset, we will wrap the environment with a :class:`minari.DataCollector` and step through it by generating actions with the path planner and waypoint controller.
#
# For this tutorial we will be using the ``PointMaze_Medium-v3`` environment to collect transition data. However, any map implementation in the PointMaze environment group can be used.
# Another important factor to take into account is that the environment is continuing, which means that it won't be ``terminated`` when reaching a goal. Instead a new goal target will be randomly selected and the agent
# will start from the location it's currently at (no ``env.reset()`` required).
#
# Lets start by importing the required modules for this tutorial:

import gymnasium as gym
import numpy as np

from minari import DataCollector, StepDataCallback


# %%
# WayPoint Planner
# ~~~~~~~~~~~~~~~~
# Our first task is to create a method that generates a trajectory to the goal in the maze.
# We have the advantage that the MuJoCo maze can be discretized into a grid of cells, which reduces
# the size of the state space. The action space for this solver will also be reduced to ``UP``, ``DOWN``, ``LEFT``,
# and ``RIGHT``. The solution trajectories will then be a set of waypoints that the agent has to follow to reach the
# goal.
# We can simply use a variation of Dynamic Programming to generate the trajectories. The method chosen in the D4RL[1] publication
# is that of Value Iteration, specifically Q-Value Iteration[2]. We will obtain the optimal Q-values by doing a series of Bellman
# updates (``50`` in total) of the form:
#
# .. math::
#   Q'(s, a) \leftarrow \sum_{s'}T(s,a,s')[R(s,a,s') + \gamma\max_{a'}Q(s',a')]
#
# **T(s,a,s')** is the transition matrix which gives the probability of reaching state **s'** when taking action **a** from state **s**.
# We consider the grid maze a deterministic space which means that if **s'** is an empty cell **T(s,a,s')** will have a value of ``1`` since
# we know that the agent will always reach that state. On the other hand, if the state **s'** is a wall the value of **T(s,a,s')** will be ``0``.
#
# Once we have the optimal Q-values (**Q***) we can generate a waypoint trajectory with the following policy:
#
# .. math::
#   \pi(s) = arg\max_{a}Q^{*}(s,a)
#
# The class below, ``QIteration``, gives access to the method ``generate_path(current_cell, goal_cell)``.  This method returns a dictionary of waypoints
# such as:
#
# .. code:: py
#
#   {(5, 1): (4, 1), (4, 1): (4, 2), (4, 2): (3, 2), (3, 2): (2, 2), (2, 2): (2, 1), (2, 1): (1, 1)}
#
# The keys of this dictionary are the current state of the agent and the values the next state of the wapoint path.

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

EXPLORATION_ACTIONS = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}


class QIteration:
    """Solves for optimal policy with Q-Value Iteration.

    Inspired by https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/q_iteration.py
    """

    def __init__(self, maze):
        self.maze = maze
        self.num_states = maze.map_length * maze.map_width
        self.num_actions = len(EXPLORATION_ACTIONS.keys())
        self.rew_matrix = np.zeros((self.num_states, self.num_actions))
        self.compute_transition_matrix()

    def generate_path(self, current_cell, goal_cell):
        self.compute_reward_matrix(goal_cell)
        q_values = self.get_q_values()
        current_state = self.cell_to_state(current_cell)
        waypoints = {}
        while True:
            action_id = np.argmax(q_values[current_state])
            next_state, _ = self.get_next_state(
                current_state, EXPLORATION_ACTIONS[action_id]
            )
            current_cell = self.state_to_cell(current_state)
            waypoints[current_cell] = self.state_to_cell(next_state)
            if waypoints[current_cell] == goal_cell:
                break

            current_state = next_state

        return waypoints

    def reward_function(self, desired_cell, current_cell):
        if desired_cell == current_cell:
            return 1.0
        else:
            return 0.0

    def state_to_cell(self, state):
        i = int(state / self.maze.map_width)
        j = state % self.maze.map_width
        return (i, j)

    def cell_to_state(self, cell):
        return cell[0] * self.maze.map_width + cell[1]

    def get_q_values(self, num_itrs=50, discount=0.99):
        q_fn = np.zeros((self.num_states, self.num_actions))
        for _ in range(num_itrs):
            v_fn = np.max(q_fn, axis=1)
            q_fn = self.rew_matrix + discount * self.transition_matrix.dot(v_fn)
        return q_fn

    def compute_reward_matrix(self, goal_cell):
        for state in range(self.num_states):
            for action in range(self.num_actions):
                next_state, _ = self.get_next_state(state, EXPLORATION_ACTIONS[action])
                next_cell = self.state_to_cell(next_state)
                self.rew_matrix[state, action] = self.reward_function(
                    goal_cell, next_cell
                )

    def compute_transition_matrix(self):
        """Constructs this environment's transition matrix.
        Returns:
          A dS x dA x dS array where the entry transition_matrix[s, a, ns]
          corresponds to the probability of transitioning into state ns after taking
          action a from state s.
        """
        self.transition_matrix = np.zeros(
            (self.num_states, self.num_actions, self.num_states)
        )
        for state in range(self.num_states):
            for action_idx, action in EXPLORATION_ACTIONS.items():
                next_state, valid = self.get_next_state(state, action)
                if valid:
                    self.transition_matrix[state, action_idx, next_state] = 1

    def get_next_state(self, state, action):
        cell = self.state_to_cell(state)

        next_cell = tuple(map(lambda i, j: int(i + j), cell, action))
        next_state = self.cell_to_state(next_cell)

        return next_state, self._check_valid_cell(next_cell)

    def _check_valid_cell(self, cell):
        # Out of map bounds
        if cell[0] >= self.maze.map_length:
            return False
        elif cell[1] >= self.maze.map_width:
            return False
        # Wall collision
        elif self.maze.maze_map[cell[0]][cell[1]] == 1:
            return False
        else:
            return True


# %%
# Waypoint Controller
# ~~~~~~~~~~~~~~~~~~~
# The step will be to create a controller to allow the agent to follow the waypoint trajectory.
# D4RL uses a PD controller to output continuous force actions from position and velocity.
# A PD controller is a variation of the PID controller often used in classical Control Theory.
# PID combines three components: a Proportial Term(P), Integral Term(I) and Derivative Term (D)
#
# 1. Proportional Term (P)
# -------------------
# The proportional term in a PID controller adjusts the control action based on the current error, which
# is the difference between the desired value (setpoint) and the current value of the process variable.
# The control action is directly proportional to the error. A higher error results in a stronger control action.
# However, the proportional term alone can lead to overshooting or instability. Note :math:`\tau` is our control value.
#
# .. math ::
#   \tau = k_{p}(\text{Error})
#
# 2. Derivative Term (D)
# -------------------
# The derivative term in a PD controller considers the rate of change of the error over time.
# It helps to predict the future behavior of the error. By dampening the control action based
# on the rate of change of the error, the derivative term contributes to system stability and reduces overshooting.
# It also helps the system respond quickly to changes in the error.
#
# .. math ::
#   \tau = k_{d}(d(\text{Error}) / dt)
#
# So for a PD controller we have the equation below. We explain what the values :math:`k_{d}` and :math:`k_{p}` mean in a bit
#
# .. math ::
#   \tau = k_{p}(\text{Error})  + k_{d}(d(\text{Error}) / dt)
#
# 3. Integral Term (I)
# -------------------
# The integral term in a PID controller integrates the cumulative error over time.
# It helps to address steady-state errors or biases that may exist in the system.
# The integral term continuously adjusts the control action based on the accumulated error,
# aiming to eliminate any long-term deviations between the desired setpoint and the actual process variable.
#
# .. math ::
#   \tau = k_{I}{\int}_0^t(\text{Error}) dt
#
# Finally for a PID controller we have the equation below
#
# .. math ::
#   \tau = k_{p}(\text{Error})  + k_{d}(d(\text{Error}) / dt) +  k_{I}\int_{0}^{t}(\text{Error}) dt
#
# In the PID controller formula, :math:`K_p`, :math:`K_i`, and :math:`K_d` are the respective gains for the proportional, integral, and derivative terms.
# These gains determine the influence of each term on the control action.
# The optimal values for these gains are typically determined through tuning, which involves adjusting
# the gains to achieve the desired control performance.
#
# Now back to our controller as stated previously, for the D4RL task we use a PD controller and we
# follow the same theme as what we have stated before as can be seen below. The :math:`Error` is equlivalent
# to the difference between the :math:`\text{goal}_\text{pose}` and :math:`\text{agent}_\text{pose}` and we replace the derivative term :math:`(d(\text{Error}) / dt)` with
# the velocity of the the agent :math:`v_{\text{agent}}`, we can think of this as a measure of the speed at which the agent
# is approaching the target position. When the agent is moving quickly towards the target, the
# derivative term will be larger, contributing to a stronger corrective action from the controller.
# On the other hand, if the agent is already close to the target and moving slowly, the derivative term will be smaller,
# resulting in a less aggressive control action.
#
# .. math ::
#   \tau = k_{p}(p_{\text{goal}} - p_{\text{agent}}) + k_{d}v_{\text{agent}}
#
# Each target position in the waypoint trajectory is converted from discrete to a continuous value and we also add some noise to
# the :math:`x` and :math:`y` coordinates to add more variance in the trajectories generated for the offline dataset.


class WaypointController:
    """Agent controller to follow waypoints in the maze.

    Inspired by https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/waypoint_controller.py
    """

    def __init__(self, maze, gains={"p": 10.0, "d": -1.0}, waypoint_threshold=0.1):
        self.global_target_xy = np.empty(2)
        self.maze = maze

        self.maze_solver = QIteration(maze=self.maze)

        self.gains = gains
        self.waypoint_threshold = waypoint_threshold
        self.waypoint_targets = None

    def compute_action(self, obs):
        # Check if we need to generate new waypoint path due to change in global target
        if (
            np.linalg.norm(self.global_target_xy - obs["desired_goal"]) > 1e-3
            or self.waypoint_targets is None
        ):
            # Convert xy to cell id
            achieved_goal_cell = tuple(
                self.maze.cell_xy_to_rowcol(obs["achieved_goal"])
            )
            self.global_target_id = tuple(
                self.maze.cell_xy_to_rowcol(obs["desired_goal"])
            )
            self.global_target_xy = obs["desired_goal"]

            self.waypoint_targets = self.maze_solver.generate_path(
                achieved_goal_cell, self.global_target_id
            )

            # Check if the waypoint dictionary is empty
            # If empty then the ball is already in the target cell location
            if self.waypoint_targets:
                self.current_control_target_id = self.waypoint_targets[
                    achieved_goal_cell
                ]
                self.current_control_target_xy = self.maze.cell_rowcol_to_xy(
                    np.array(self.current_control_target_id)
                )
            else:
                self.waypoint_targets[
                    self.current_control_target_id
                ] = self.current_control_target_id
                self.current_control_target_id = self.global_target_id
                self.current_control_target_xy = self.global_target_xy

        # Check if we need to go to the next waypoint
        dist = np.linalg.norm(self.current_control_target_xy - obs["achieved_goal"])
        if (
            dist <= self.waypoint_threshold
            and self.current_control_target_id != self.global_target_id
        ):
            self.current_control_target_id = self.waypoint_targets[
                self.current_control_target_id
            ]
            # If target is global goal go directly to goal position
            if self.current_control_target_id == self.global_target_id:
                self.current_control_target_xy = self.global_target_xy
            else:
                self.current_control_target_xy = (
                    self.maze.cell_rowcol_to_xy(
                        np.array(self.current_control_target_id)
                    )
                    - np.random.uniform(size=(2,)) * 0.2
                )

        action = (
            self.gains["p"] * (self.current_control_target_xy - obs["achieved_goal"])
            + self.gains["d"] * obs["observation"][2:]
        )
        action = np.clip(action, -1, 1)

        return action


# %%
# Modified StepDataCallback
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We will also need to create our own custom callback function to record the data after each step.
# As previously mentioned, the task is continuing and the environment won't be ``terminated`` or ``truncated`` when reaching a goal.
# Instead a new target will be randomly selected in the map and the agent will continue stepping to this new goal. For this reason, to divide the dataset into episodes,
# we will have to truncate the dataset ourselves when a new goal is reached. This can be done by overriding the ``'truncations'`` key in the step data return when the
# agent returns ``success=True`` in the ``'infos'`` item.
#
# In the :class:`minari.StepDataCallback` we can add new keys to infos that we would also want to save in our Minari dataset. For example in this
# case we will be generating new hdf5 datasets ``qpos``, ``qvel``, and ``goal`` in the ``infos`` subgroup of each episode group.
#


class PointMazeStepDataCallback(StepDataCallback):
    """Add environment state information to 'infos'.

    Also, since the environment generates a new target every time it reaches a goal, the environment is
    never terminated or truncated. This callback overrides the truncation value to True when the step
    returns a True 'succes' key in 'infos'. This way we can divide the Minari dataset into different trajectories.
    """

    def __call__(
        self, env, obs, info, action=None, rew=None, terminated=None, truncated=None
    ):
        qpos = obs["observation"][:2]
        qvel = obs["observation"][2:]
        goal = obs["desired_goal"]

        step_data = super().__call__(env, obs, info, action, rew, terminated, truncated)

        if step_data["info"]["success"]:
            step_data["truncation"] = True
        step_data["info"]["qpos"] = qpos
        step_data["info"]["qvel"] = qvel
        step_data["info"]["goal"] = goal

        return step_data


# %%
# Collect Data and Create Minari Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now we will finally perform our data collection and create the Minari dataset. This is as simple as wrapping the environment with
# the :class:`minari.DataCollector` wrapper and add the custom callback methods. Once we've done this we can step the environment with the ``WayPointController``
# as our policy. For the tutorial, we collect 10,000 transitions. Thus, we initialize the environment with ``max_episode_steps=10,000`` since that's the total amount of steps we want to
# collect for our dataset and we don't want the environment to get ``truncated`` during the data collection due to a time limit.
#


dataset_name = "pointmaze/umaze-v0"
total_steps = 10_000

# continuing task => the episode doesn't terminate or truncate when reaching a goal
# it will generate a new target. For this reason we set the maximum episode steps to
# the desired size of our Minari dataset (evade truncation due to time limit)
env = gym.make("PointMaze_Medium-v3", continuing_task=True, max_episode_steps=total_steps)

# Data collector wrapper to save temporary data while stepping. Characteristics:
#   * Custom StepDataCallback to add extra state information to 'infos' and divide dataset in different episodes by overridng
#     truncation value to True when target is reached
#   * Record the 'info' value of every step
collector_env = DataCollector(
    env, step_data_callback=PointMazeStepDataCallback, record_infos=True
)

obs, _ = collector_env.reset(seed=123)

waypoint_controller = WaypointController(maze=env.maze)

for n_step in range(int(total_steps)):
    action = waypoint_controller.compute_action(obs)
    # Add some noise to each step action
    action += np.random.randn(*action.shape) * 0.5
    action = np.clip(
        action, env.action_space.low, env.action_space.high, dtype=np.float32
    )

    obs, rew, terminated, truncated, info = collector_env.step(action)

dataset = collector_env.create_dataset(
    dataset_id=dataset_name,
    algorithm_name="QIteration",
    code_permalink="https://github.com/Farama-Foundation/Minari/blob/main/docs/tutorials/dataset_creation/point_maze_dataset.py",
    author="Rodrigo Perez-Vicente",
    author_email="rperezvicente@farama.org",
)


# %%
# References
# ~~~~~~~~~~
#
# [1] Fu, Justin, et al. ‘D4RL: Datasets for Deep Data-Driven Reinforcement Learning’.
# CoRR, vol. abs/2004.07219, 2020, https://arxiv.org/abs/2004.07219..
#
# [2] Lambert, Nathan. ‘Fundamental Iterative Methods of Reinforcement Learnin’.
# Apr 8, 2020, https://towardsdatascience.com/fundamental-iterative-methods-of-reinforcement-learning-df8ff078652a
