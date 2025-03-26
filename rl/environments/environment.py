from rl.state import State
from utils.util import stdout
import abc
from network.socket_communication.signal_server import SignalServer
from network.socket_communication.reward_client import ValueSender


class Environment(abc.ABC):
    '''
    Basic class for an environment of a learning :py:class:`rl.agent.Agent`
    '''

    def __init__(self, learning_problem:str, output_color_code:str, rl_agent, pos_reward_sender:ValueSender=None, neg_reward_sender:ValueSender=None, bad_r_thres=0.0):
        from config.settings import EPISODE_COUNTER, MAX_LEN_SIGNALS
        from config.settings import QUEUE_CONFIG, SIGNALS,IS_SIMULATION

        self.learning_problem=learning_problem
        self.id=rl_agent.id
        self.output_color_code=output_color_code
        self.rl_agent= rl_agent
        rl_agent.env=self

        if not IS_SIMULATION:
            self.signal_server = SignalServer.get_instance(QUEUE_CONFIG, *SIGNALS)

        self.episode_counter=EPISODE_COUNTER
        self.pos_reward_sender=pos_reward_sender
        self.neg_reward_sender = neg_reward_sender
        self.bad_r_thres=bad_r_thres

        self.next_episode()


    def next_episode(self):
        '''
        Set a new episode for the environment.

        :return:
        '''
        from config.settings import Output, SHOW_RL_LOG

        self.init_vars()
        self.episode_counter += 1

        self.rl_agent.next_episode()
        self.rl_agent.action, output=self.rl_agent.select_action(self.state)
        self.rl_agent.exec_action(self.state)

        if SHOW_RL_LOG:

            print()
            print()
            stdout.indent_print_color(self.output_color_code, Output.NEW_EPISODE.value)
            stdout.indent_print_color(self.output_color_code, str.format("Problem: {0}, Env-ID: {1}, Agent-ID: {2}", self.learning_problem, self.id, self.rl_agent.id))
            stdout.indent_print_color(self.output_color_code, str.format("episode {0}, step {1}", self.episode_counter, self.step_counter))

            stdout.indent_print_color(self.output_color_code,
                str.format(" execute in step {0}: (s,a): {1}, {2} = {3}", self.step_counter, self.state, self.rl_agent.action, output),
                indent_level=2, indent_symbol=".")

    def reset(self):
        from config.settings import EPISODE_COUNTER
        self.episode_counter=EPISODE_COUNTER

        self.init_vars()
        #self.init_vars()
        self.rl_agent.reset()

    def init_vars(self):
        from config.settings import STEP_COUNTER
        self.step_counter = STEP_COUNTER

        self.reward = float(0.0)
        self.last_state = None
        self.state = self.get_init_state()

    def process_rl_step(self):
        '''
        Method for processing one rl step.

        Precondition: a chosen
        Step:        (r)-->(a')
        :return:
        '''
        from config.settings import SHOW_RL_LOG


        self.step_counter += 1
        state=self.state.copy()
        action = self.rl_agent.action.copy()

        # observe s', r
        self.modify_environment(self.state, action)
        reward = self.get_reward()

        if SHOW_RL_LOG:
            print('')
            print('')
            stdout.indent_print_color(self.output_color_code, str.format("Problem: {0}, Env-ID: {1}, Agent-ID: {2}", self.learning_problem, self.id, self.rl_agent.id))
            stdout.indent_print_color(self.output_color_code, str.format("Episode {0}, step {1}", self.episode_counter, self.step_counter))
            stdout.indent_print_color(self.output_color_code,
                str.format(" update in step {0}: (s,a,r): {1}, {2}, {3}", (self.step_counter - 1), state, action,
                           reward),
                indent_level=2, indent_symbol=".")

        self.apply_reward(self.last_state, action, self.state, reward=reward)

        self.rl_agent.action, output = self.rl_agent.select_action(self.state)
        self.rl_agent.exec_action(self.state)


        if SHOW_RL_LOG:
            stdout.indent_print_color(self.output_color_code,
                str.format(" execute in step {0}: (s,a): {1}, {2} = {3}", self.step_counter, self.state, self.rl_agent.action, output),
                indent_level=2, indent_symbol=".")

            self.rl_agent.print_knowledge(self.output_color_code)


    def apply_reward(self, last_state, last_action, state, reward: float):
        '''
        Update learning model and history information of agent.
        :param reward:
        :return:
        '''

        self.rl_agent.update(last_state, last_action, state, reward=reward)
        self.rl_agent.add_step_information(last_state, last_action, reward)

    def modify_environment(self, state: State, action: 'Action'):
        '''
        State update for last_state and state.

        In most cases it's sufficient to just get the state from get_state().
        However a special modelling can require a state modification according to current state and given action

        :param last_state:
            Current state
        :param action:
        :return:
        '''
        self.last_state=state.copy()
        self.state=self.modify_state(state.copy(), action.copy())

    @abc.abstractmethod
    def modify_state(self, state: State, action: 'Action') -> State:
        '''
        This method builds a state from the environment.

        :param init_state:
            Indicates, whether an initial state should be returned
        :return: State
            New State
        '''
        pass

    @abc.abstractmethod
    def get_init_state(self) -> State:
        '''
        This method builds a initial state from the environment.

        :param init_state:
            Indicates, whether an initial state should be returned
        :return: State
            New State
        '''
        pass

    @abc.abstractmethod
    def get_reward(self) -> float:
        pass


    def send_reward(self, reward):
        # send reward to interested client
        if self.pos_reward_sender is not None and self.neg_reward_sender is not None:
            if reward <= self.bad_r_thres:
                self.neg_reward_sender.send_reward(reward)
            elif reward > self.bad_r_thres:
                self.pos_reward_sender.send_reward(reward)

