import abc
import numpy as np
import json
from os import path
from utils.util import stdout
from utils.test_package import synchronisation
from rl.actions import Action
from rl.state import State, VectorConverterState

import random
from utils.util.math.basis import Basis
from rl.logging.sar import SAR

from config.settings import *
from utils.util.math.basis import FourierBasis
from rl.physical_handler import PhysicalAgent




def build_agent(algorithm: 'Algorithm', algorithm_type: 'AlgorithmType', selection_strategy:'SelectionStrategies', selection_config:float,env:'Environment', physical_agent: PhysicalAgent):
    '''
    Method for constructing a learning agent

    :param algorithm: Algorithm
        Algorithm, according to which the agent learns
    :param gradient_type: GradientType
        Gradient type of the gradient algorithm
    :param env: Environment
        Learning environment for the agent
    :return:
    '''

    from config.settings import Algorithm, DEFAULT_Q_VALUE,ALPHA,GAMMA,LAMBDA,BASIS, NUM_ACTIONS
    agent = None
    physical_agent=physical_agent

    if algorithm == Algorithm.BANDIT or algorithm == Algorithm.Q:
        agent = RLAgentQ(algorithm=algorithm,
                         algorithm_type=algorithm_type,
                         selection_strategy=selection_strategy,
                         env=env,
                         physical_agent=physical_agent,

                         alpha=ALPHA,
                         selection_config=selection_config,
                         gamma= GAMMA,
                         reset_episode=True,
                         default_q_value=DEFAULT_Q_VALUE)
        # dim=dimension of feature vector
    elif algorithm == Algorithm.GRADIENT:
        agent = RLAgentGradient(algorithm=algorithm,
                                algorithm_type=algorithm_type,
                                selection_strategy=selection_strategy,
                                env=env,
                                physical_agent=physical_agent,

                                alpha=ALPHA,
                                beta=BETA,
                                selection_config=selection_config,
                                gamma= GAMMA,
                                reset_episode=True,

                                num_actions=NUM_ACTIONS,
                                basis=BASIS,
                                _lambda=LAMBDA,
                                q_accuracy=VALUE_ACCURACY)
    elif algorithm == Algorithm.RANDOM:
        #agent = RLAgentRandom(env,physical_agent)
        pass

    return agent


##########################################################################################################


class Agent(abc.ABC, synchronisation.Observable):
    '''
    Basic class for a learning agent
    '''
    __metaclass__ = abc.ABCMeta
    num_existing_agents=0

    def __init__(self, algorithm: 'Algorithm', algorithm_type:'AlgorithmType', selection_strategy:'SelectionStrategies', env: 'Environment', physical_agent: 'PhysicalAgent',
                 alpha:float, selection_config: float,
                 reset_episode:bool):
        from config.settings import RL_PATH
        synchronisation.Observable.__init__(self=self)

        Agent.num_existing_agents= Agent.num_existing_agents + 1
        self.id=Agent.num_existing_agents
        self.algorithm = algorithm
        self.algorithm_type=algorithm_type
        self.selection_strategy=selection_strategy
        self.env = env
        self.physical_agent=physical_agent

        self.alpha=alpha
        self.selection_config=selection_config

        self.reset_episode=reset_episode

        self.avg_rewards_episode = []
        self.avg_rewards_cumulative = []
        self.avg_rewards_deviated = []

        self.action=None

        # sar_history is a 2-D list: for each episode it contains all steps
        self.sar_history=[]
        self._path = path.abspath(path.join(path.dirname(__file__), RL_PATH))

    def next_episode(self):
        '''
        Set a new episode for the agent

        :return:
        '''
        self.avg_rewards_episode = []
        self.sar_history.append([])

    def set_env(self, env):
        self.env=env

    def reset(self):
        '''
        Total reset of the agent

        :return:
        '''
        self.avg_rewards_episode = []
        self.avg_rewards_cumulative = []
        self.avg_rewards_deviated = []
        self.sar_history = []
        self.action = None

    # state vs no state function
    def exec_action(self, state:State):
        state = state.copy()
        action = self.action.copy()
        action.apply(state, self.physical_agent)   #TODO: beim bandit evtl None==state

    @abc.abstractmethod
    def save_policy(self, _path: str, safe_file: str):
        '''
        Saves the knowledge of the agent to a specified file.

        :param _path:str
            Path to the file
        :param safe_file:str
            File name
        :return:
        '''
        pass

    @abc.abstractmethod
    def load_policy(self, _path: str, safe_file: str):
        '''
        Loads the knowledge of the agent from a specified file.

        :param _path:str
            Path to the file
        :param safe_file:str
            File name
        :return:
        '''
        pass

    @abc.abstractmethod
    def update(self, *args):
        """
        Update the actual knowledge.

        :param action: Action
            Action applied
        :param reward: float
            Reward gained by applying action
        :return:
        """
        pass

    @abc.abstractmethod
    def select_action(self, *args) -> Action:
        pass

    @abc.abstractmethod
    def _select_random_action(self, *args) -> Action:
        '''
        Among available actions select a random one.

        :param args:
        :return: Action
            Action that has been chosen
        '''
        pass

    @abc.abstractmethod
    def _select_max_action(self, *args) -> Action:
        '''
        Among available actions select the one, that has the maximum outcome regarding the actual knowledge

        :param args:
        :return: Action
            Action that has been chosen
        '''
        pass

    @abc.abstractmethod
    def add_step_information(self, *args):
        '''
        Update the agents history information for logging and plotting purposes.

        :param args:
        :return:
        '''
        pass

    @abc.abstractmethod
    def print_knowledge(self, color_code:str):
        '''
        Print the knowledge of the agent

        :param args:
        :return:
        '''
        pass


    def _add_reward_information(self, reward: float):
        '''
        This method updates information for plotting: avg_rewards_episode, avg_rewards_cumulative, avg_rewards_deviated

        :param reward:
        :return:
        '''

        # averages of episode (avg: steps[0 to i-1] (single episode),  step[i])
        if len(self.avg_rewards_episode) > 0:
            step = len(self.avg_rewards_episode) + 1
            self.avg_rewards_episode.append((self.avg_rewards_episode[-1] * (step-1) + reward)/(step))
        else:
            self.avg_rewards_episode.append(reward)

        # cumulative averages of episode (avg: step[i] of all episodes, step[i] )
        if len(self.avg_rewards_cumulative) >= len(self.avg_rewards_episode):
            index = len(self.avg_rewards_episode) - 1
            self.avg_rewards_cumulative[index] = \
                (
                        (self.avg_rewards_cumulative[index] * (self.env.episode_counter - 1) + self.avg_rewards_episode[-1]
                         )/self.env.episode_counter
                 )
        # just for first episode
        else:
            self.avg_rewards_cumulative.append(self.avg_rewards_episode[-1])

        # deviated averages over all episodes (array (all steps 0-i) of arrays ( over all episodes: all avg_rewards_episode of step i))
        if len(self.avg_rewards_deviated) >= len(self.avg_rewards_episode):
            index = len(self.avg_rewards_episode) - 1
            self.avg_rewards_deviated[index].append(self.avg_rewards_episode[-1])
        else:
            # add new array for current step
            self.avg_rewards_deviated.append([])
            # add average for last step
            self.avg_rewards_deviated[-1].append(self.avg_rewards_episode[-1])
        # notify observers
        self.set_changed()
        self.notify_observers(self.avg_rewards_deviated)


##########################################################################################################


class StateBasedAgent(Agent):
    def __init__(self, algorithm: 'Algorithm', algorithm_type:'AlgorithmType', selection_strategy:'SelectionStrategies', env: 'Environment', physical_agent: 'PhysicalAgent',
                 alpha: float, selection_config:float, gamma: float,
                 reset_episode: bool):
        super().__init__(algorithm, algorithm_type, selection_strategy, env, physical_agent, alpha, selection_config, reset_episode)

        self.gamma = gamma

    def select_action(self, state:State) -> (Action, str):
        state = state.copy()
        action = None
        output = ' '

        if self.selection_strategy == SelectionStrategies.EPSILON_GREEDY:
            output += SelectionStrategies.EPSILON_GREEDY.value + ' '
            rnd = random.random()

            if rnd < self.selection_config:
                output += "random action"
                action = self._select_random_action(state)
            else:
                output += "max action"
                action = self._select_max_action(state)

        elif self.selection_strategy == SelectionStrategies.UCB:
            output += SelectionStrategies.UCB.value + ' '
            # TODO: implement
        elif self.selection_strategy == SelectionStrategies.SOFTMAX:
            output += SelectionStrategies.SOFTMAX.value + ' '
            # TODO: implement

        return (action, output)

    def _select_random_action(self, state: State) -> Action:
        num_actions = len(state.actions)
        rnd = random.randint(0, num_actions - 1)
        return state.actions[rnd]

    def _select_max_action(self, state: State) -> Action:
        a_max = []
        val_max = -float('inf')
        # print()
        # print(state)

        for action in state.actions:
            q_value = self.get_value(state, action)
            #print("Action: " + str(action) + "=" + str(q_value))
            if q_value > val_max:
                val_max = q_value
                a_max = [action]
            elif q_value == val_max:
                a_max.append(action)
        if len(a_max) > 0:
            rnd = random.randint(0, len(a_max) - 1)
            return a_max[rnd]
        if SHOW_RL_LOG:
            print('RLAgentQ: picking max action failed-->random action')
        return self._select_random_action(state)


    @abc.abstractmethod
    def update(self, last_state: State, last_action: Action, state: State, reward: float):
        '''
        Update the actual knowledge.

        :param state: State
            Current state
        :param sprime: State
            Next state
        :param action:
            Action applied in state
        :param reward:
            Reward gained by applying action in state
        :return:
        '''

        pass

    @abc.abstractmethod
    def get_value(self, state: State, action: Action) -> float:
        """
        Q-value for specified state - action pair

        :param state: State
            Current state
        :param action: : Action
            Action applied in state
        :return: float
            q value as float value
        """
        pass

    def add_step_information(self, state: State, action: Action, reward: float):
        #TODO: implement in plot class
        #self._add_reward_information(reward)
        # add information to the history of the current episode
        self.sar_history[-1].append(SAR(state, action, reward))


    '''
    returns list of all sar information of all unique visited (s,a)-pairs
    '''
    def get_history_information(self):
        '''
        Collect all visited (s,a)-pairs from the whole history (over all episodes)

        :return: list[SAR]
            List of all sar information of all unique visited (s,a)-pairs
        '''
        from copy import deepcopy
        unique_sa = []
        if self.reset_episode:
            sar_history=[deepcopy(self.sar_history[-1])]
        else:
            sar_history=deepcopy(self.sar_history)

        for sar_episode in sar_history:
            for sar in sar_episode:
                if not (sar in unique_sa):
                    unique_sa.append(sar)
        #sorted(unique_sa, key=lambda x: (x[0], -x[1]))
        sorted(unique_sa, key=lambda x: (x.state.hash_id, x.action.hash_id), reverse=True)
        return unique_sa


    def print_knowledge(self, color_code:str):
        stdout.indent_print_color(color_code, Output.Q_VALUE_PRE.value)

        sar_list=self.get_history_information()

        # print all visited (s,a)-pairs
        for sar in sar_list:
            action=sar.get_action()
            knowledge=str.format("q({0}, {1}) = {2:.2f}", sar.get_state(), action,
                                 self.get_value(sar.get_state(), action))
            if self.algorithm==Algorithm.BANDIT:
                if action in self.n_table:
                    n=self.n_table[action]
                else:
                    n=0
                knowledge += str.format("  --  n({0}) = {1}", action,n)
            stdout.indent_print_color(color_code, knowledge)
        stdout.indent_print_color(color_code, Output.Q_VALUE_SUF.value)


    def get_Q(self, states:[State]):
        import copy
        q_approximated={}
        for state in states:
            q_approximated[state] = {}
            for action in state.actions:
                q_approximated[state][action] = self.get_value(state, action)
        return copy.deepcopy(q_approximated)

##########################################################################################################

class RLAgentQ(StateBasedAgent):

    def __init__(self, algorithm: 'Algorithm', algorithm_type:'AlgorithmType', selection_strategy:'SelectionStrategies', env: 'Environment', physical_agent: 'PhysicalAgent',
                 alpha: float, selection_config: float, gamma: float,
                 reset_episode: bool=False, default_q_value:float=0.0):


        super().__init__(algorithm, algorithm_type, selection_strategy, env, physical_agent, alpha, selection_config, gamma, reset_episode)
        self.default_q_value = default_q_value


    def init_q_table(self):
        '''
        Initializes the q table and n table of the k armed bandit.
        :return:
        '''
        self.q_table = {}   # table for knowledge

        if self.algorithm == Algorithm.BANDIT:
            self.q_table[self.env.__class__.EMPTY_STATE]={}
            self.n_table = {}
            a=self.env.__class__.ACTIONS
            for action in self.env.__class__.ACTIONS:
                self.q_table[self.env.__class__.EMPTY_STATE][action]=self.default_q_value
                self.n_table[action] = 0   # table for number of times an action has been chosen


    def reset(self):
        super().reset()
        self.init_q_table()

    def next_episode(self):
        super().next_episode()#
        if self.reset_episode:
            self.init_q_table()

    def save_policy(self, _path: str, safe_file: str):
        pass

    def load_policy(self, _path: str, safe_file: str):
        raise Exception("Function Not implemented")
        pass

    def update(self, last_state: State, last_action: Action, state: State, reward: float):

        old_val=self.get_value(last_state, last_action)

        if self.algorithm == Algorithm.BANDIT:
            if not last_action in self.n_table:
                self.n_table[last_action] = 0
            self.n_table[last_action] = self.n_table[last_action] + 1
            new_val = old_val + (float(1.0) / self.n_table[last_action]) * (reward - old_val)

        elif self.algorithm == Algorithm.Q:
            if self.algorithm_type==QType.STANDARD:
                max_action=self._select_max_action(state.copy())
                new_val = old_val + self.alpha * (reward + self.gamma * self.get_value(state, max_action) - old_val)

        if last_state in self.q_table:
            # update q-value for action
            self.q_table[last_state][last_action] = new_val
        else:
            # add state-action pair and value toq_table
            self.q_table[last_state] = {}
            self.q_table[last_state][last_action] = new_val


    def get_value(self, state: State, action: Action) -> float:
        if state in self.q_table:
            if action in self.q_table[state]:
                return self.q_table[state][action]
        return self.default_q_value


##########################################################################################################


class RLAgentGradient(StateBasedAgent):

    def __init__(self, algorithm: 'Algorithm', algorithm_type:'AlgorithmType', selection_strategy:'SelectionStrategies', env: 'StateBasedEnvironment',
                 physical_agent: 'PhysicalAgent',
                 alpha: float, beta:float, selection_config: float, gamma: float,
                 reset_episode: bool,
                 num_actions: int, basis: Basis, _lambda: float, q_accuracy: int):

        # avoid divergence by modifying alpha
        super().__init__(algorithm, algorithm_type, selection_strategy, env, physical_agent, alpha , selection_config, gamma, reset_episode)

        self.beta=beta
        self.basis = basis
        self._lambda = _lambda
        self.size = basis.dimension()       # in case of fourier == number of coefficients == hence: k
        self.num_actions = num_actions
        self.q_accuracy = q_accuracy
        self.init_vectors()

    def reset(self):
        super().reset()
        self.init_vectors()

    def init_vectors(self):
        # generate weights, traces, theta matrix
        self._weights = np.full(fill_value=1.5/self.size, shape=(self.num_actions, self.size), dtype=float)
        self._traces = np.zeros(shape=(self.num_actions, self.size), dtype=float)
        self._theta = np.ones(shape=(self.num_actions, self.size), dtype=float)

    def next_episode(self):
        super().next_episode()#
        if self.reset_episode:
            self.init_vectors()

    def save_policy(self, _path: str, safe_file: str):
        for ending in ["_weights", "_traces", "_theta"]:
            file = path.join(_path, str.format("{0}{1}", safe_file, ending))
            with open(file, 'w') as f:
                json.dump(getattr(self, str(ending)).tolist(), f)

    def load_policy(self, _path: str, safe_file: str):
        stdout.indent_print("Gradient Agent: Strategy loaded")
        for ending in ["_weights", "_traces", "_theta"]:
            file = path.join(_path, str.format("{0}{1}", safe_file, ending))
            with open(file, 'r') as f:
                setattr(self, str(ending), np.array(json.load(f)))

    def update(self, last_state: VectorConverterState, last_action: Action, state: VectorConverterState, reward: float):
        from copy import deepcopy
        # q value of current action and current state
        q_last_s_a = self.get_value(last_state, last_action)

        # q value of max action in next state
        max_action_s = self._select_max_action(state)
        q_s_a = self.get_value(state, max_action_s)

        # feature matrix for last state
        feature_matrix_last_s_a = np.zeros(shape=(self.num_actions, self.size))
        feature_matrix_last_s_a[last_action.hash_id] = self.basis.convert_features(last_state.get_feature_vector())

        # feature vector for state
        feature_vector_s = np.asarray(self.basis.convert_features(state.get_feature_vector()))

        # reward delta
        # r + gamma * q_new - q_old
        delta = reward + self.gamma * q_s_a - q_last_s_a

        if self.algorithm_type == GradientType.STANDARD:
            # formula: alpha * delta * phi_last_s_a
            update_matrix = (self.alpha * delta) * feature_matrix_last_s_a
            self._weights += update_matrix


        elif self.algorithm_type == GradientType.TDC:
            # formula: w + alpha*delta*phi_last_s_a    +    alpha*gamma* phi_s_max_a * val_theta_last_s_a
            # val_theta_last_s_a = phi_last_s_a * theta_last_a
            val_theta_last_s_a = (
                np.dot(feature_matrix_last_s_a[last_action.hash_id], self._theta[last_action.hash_id]))

            update_matrix = deepcopy((self.alpha * delta * feature_matrix_last_s_a))  # first part
            update_vector = (self.alpha * self.gamma * feature_vector_s * val_theta_last_s_a)  # second part
            update_matrix[last_action.hash_id] = update_matrix[last_action.hash_id] - update_vector  # first-second part

            self._weights += update_matrix

            # formula (1 - alpha) * (delta - v) * phi
            self._theta += ((self.beta) * (delta - val_theta_last_s_a) * feature_matrix_last_s_a)

        elif self.algorithm_type == GradientType.GTD2:
            # formula: w + alpha*     (phi_last_s_a    +    gamma* phi_s_max_a)   * val_theta_last_s_a
            # val_theta_last_s_a = phi_last_s_a * theta_last_a
            val_theta_last_s_a = (
                np.dot(feature_matrix_last_s_a[last_action.hash_id], self._theta[last_action.hash_id]))
            # formula: w + alpha * (phi - gamma + phi') * v

            # calculate inner matrix of formula
            update_matrix = deepcopy(feature_matrix_last_s_a)
            update_matrix[last_action.hash_id] = feature_matrix_last_s_a[last_action.hash_id] - (
                        self.gamma * feature_vector_s)

            # weights += alpha (inner_matrix) * v
            self._weights += (self.alpha * update_matrix * val_theta_last_s_a)

            # formula (1 - alpha) * (delta - v) * phi
            self._theta += ((self.beta) * (delta - val_theta_last_s_a) * feature_matrix_last_s_a)


        elif self.algorithm_type == GradientType.ACC_TRACE:
            # formula lambda * gamma * e + phi
            self._traces = self._traces * (self._lambda * self.gamma) + feature_matrix_last_s_a
            # formula: alpha * delta * e
            self._weights += (self.alpha * delta) * self._traces

        elif self.algorithm_type == GradientType.REP_TRACE:
            # formula: max(lambda * gamma * e, phi)
            self._traces = np.maximum(self._lambda * self.gamma * self._traces, feature_matrix_last_s_a)
            # formula: alpha * delta * e
            self._weights += (self.alpha * delta) * self._traces


    def get_value(self, state: VectorConverterState, action: Action) -> float:
        """
        Get q-value for given state-action pair
        :param state: State
        :param action: Action
        :return:
        """
        # dot product of feature vector and the weight-matrix row at the index for given action
        # test=state.get_feature_vector()
        # a=self._weights[action.hash_id]
        return np.dot(self.basis.convert_features(state.get_feature_vector()), self._weights[action.hash_id]).round(self.q_accuracy)
