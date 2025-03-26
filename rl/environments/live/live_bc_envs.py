from rl.actions import BasicAction
from rl.state import BCState, State
from utils.social_sensing.processing import SocialSignalProcessor
import copy
from utils.sw_patterns.behavioral.observer import ConcreteObserver
from network.socket_communication.signal_server import SignalEvents
from rl.environments.environment import Environment
from network.socket_communication.reward_client import ValueSender


class BCEnvironment(Environment, ConcreteObserver):
    ACTIONS=[
        BasicAction(0,'wdh_down'),
        BasicAction(1,'wdh_up'),
        BasicAction(2,'lex_down'),
        BasicAction(3,'lex_up'),
        BasicAction(4, 'nod_down'),
        BasicAction(5, 'nod_up'),
        BasicAction(6, 'stay')
    ]
    EMPTY_STATE = BCState([
        BasicAction(1,'wdh_down'),

        BasicAction(2,'lex_down'),
        BasicAction(3,'lex_up'),
        BasicAction(4, 'nod_down'),
        BasicAction(5, 'nod_up'),
        BasicAction(6, 'stay')
    ], [1,  0,1,0, 0,1,0])

    '''
    This class extends :class:`StateBasedEnvironment` with additional information for social signals.
    '''
    def __init__(self, learning_problem:str, output_color_code:str, rl_agent, pos_reward_sender:ValueSender=None, neg_reward_sender:ValueSender=None, bad_r_thres=0.0):
        from config.settings import MAX_LEN_SIGNALS, IS_SIMULATION

        self.old_valence = float(0.0)
        self.signal_processor=SocialSignalProcessor()

        Environment.__init__(self, learning_problem, output_color_code, rl_agent, pos_reward_sender, neg_reward_sender, bad_r_thres)

        if not IS_SIMULATION:
            ConcreteObserver.__init__(self, [(self.signal_server, SignalEvents.SPOKEN)])

    def get_init_state(self)->State:
        return self.modify_state(None, None)

    def modify_state(self, state: BCState, action: 'Action') -> State:
        '''
         By the help of this method the rl component can pick up a state for the corresponding timestep.

         :param init_state:
             Indicates, whether information for an initial state should be returned (useless because of the social signals)
         :return: (StateNames, [int])
         '''

        from config.settings import Algorithm, STATE_REMAINING_NUMBER, valence_user_key, arousal_user_key
        if state is None and action is None:
            return BCEnvironment.EMPTY_STATE


        new_state_entries = copy.deepcopy(state.get_state_vector())
        #stay
        if action.name=='stay':
            return copy.deepcopy(state)

        #modify wdh
        elif action.name == 'wdh_down':
            if  new_state_entries[0]==1:
                new_state_entries[0]=0

        elif action.name == 'wdh_up':
            if  new_state_entries[0]==0:
                new_state_entries[0]=1

        #modify lex style
        elif action.name == 'lex_down':
            if new_state_entries[1]==1:
                new_state_entries[1:4] = [1, 0, 0]
            elif new_state_entries[2]==1:
                new_state_entries[1:4] = [1, 0, 0]
            elif new_state_entries[3]==1:
                new_state_entries[1:4] = [0, 1, 0]
        elif action.name == 'lex_up':
            if new_state_entries[1]==1:
                new_state_entries[1:4] = [0, 1, 0]
            elif new_state_entries[2]==1:
                new_state_entries[1:4] = [0, 0, 1]
            elif new_state_entries[3]==1:
                new_state_entries[1:4] = [0, 0, 1]

         #modify nod style
        elif action.name == 'nod_down':
            if new_state_entries[4]==1:
                new_state_entries[4:7] = [1, 0, 0]
            elif new_state_entries[5]==1:
                new_state_entries[4:7] = [1, 0, 0]
            elif new_state_entries[6]==1:
                new_state_entries[4:7] = [0, 1, 0]
        elif action.name == 'nod_up':
            if new_state_entries[4]==1:
                new_state_entries[4:7] = [0, 1, 0]
            elif new_state_entries[5]==1:
                new_state_entries[4:7] = [0, 0, 1]
            elif new_state_entries[6]==1:
                new_state_entries[4:7] = [0, 0, 1]


        new_actions = self.get_actions_from_entry(new_state_entries)
        new_state = BCState(actions=new_actions, entries=new_state_entries)

        return new_state

    def get_reward(self) -> float:
        from config.settings import valence_user_key, arousal_user_key

        val_list=list(self.signal_server.SERVER_QUEUES[valence_user_key])
        arousal_list=list(self.signal_server.SERVER_QUEUES[arousal_user_key])
        explicit_feedback_queue = self.signal_server.SERVER_QUEUES['explicit_feedback']

        # for explicit feedback only, if signal lists are empty
        if len(explicit_feedback_queue)>0:
            reward=explicit_feedback_queue[-1]
            self.send_reward(reward)
            return reward

        new_val, new_arousal=self.signal_processor.mean_valence_arousal(val_list,arousal_list)

        reward=new_val-self.old_valence
        self.old_valence=new_val
        self.send_reward(reward)

        return reward


    def get_actions_from_entry(self, state_entry:[]):
        new_actions = [copy.deepcopy(BCEnvironment.ACTIONS[-1])] #init with stay

        #wdh actions
        if state_entry[0]==0:
            new_actions.append(copy.deepcopy(BCEnvironment.ACTIONS[1]))
        elif state_entry[0]==1:
            new_actions.append(copy.deepcopy(BCEnvironment.ACTIONS[0]))

        #lex actions
        if state_entry[1:4]==[1, 0, 0]:
            new_actions.append(copy.deepcopy(BCEnvironment.ACTIONS[3])) #nur up
        elif state_entry[1:4]==[0, 1, 0]:
            new_actions.append(copy.deepcopy(BCEnvironment.ACTIONS[2]))
            new_actions.append(copy.deepcopy(BCEnvironment.ACTIONS[3]))
        elif state_entry[1:4]==[0, 0, 1]:
            new_actions.append(copy.deepcopy(BCEnvironment.ACTIONS[2])) #nur down

        #nod actions
        if state_entry[4:7]==[1, 0, 0]:
            new_actions.append(copy.deepcopy(BCEnvironment.ACTIONS[5])) #nur up
        elif state_entry[4:7]==[0, 1, 0]:
            new_actions.append(copy.deepcopy(BCEnvironment.ACTIONS[4]))
            new_actions.append(copy.deepcopy(BCEnvironment.ACTIONS[5]))
        elif state_entry[4:7]==[0, 0, 1]:
            new_actions.append(copy.deepcopy(BCEnvironment.ACTIONS[4])) #nur down

        return new_actions



    def update(self, event):
        if event==SignalEvents.EXPLICIT_REWARD or event==SignalEvents.SPOKEN:
            self.process_rl_step()







class BCEnvironmentBandit(Environment, ConcreteObserver):
    ACTIONS=[
        # 0  000 000
        # first bit: low/high
        # 3 bits: no/low/high
        # 3 bits: no/low/high

        # First bit low
        BasicAction(0,'LNN'),
        BasicAction(1,'LNL'),
        BasicAction(2,'LNH'),

        BasicAction(3,'LLN'),
        BasicAction(4, 'LLL'),
        BasicAction(5, 'LLH'),

        BasicAction(6, 'LHN'),
        BasicAction(7, 'LHL'),
        BasicAction(8, 'LHH'),
        ##############################
        #first bit high
        BasicAction(9, 'HNN'),
        BasicAction(10, 'HNL'),
        BasicAction(11, 'HNH'),

        BasicAction(12, 'HLN'),
        BasicAction(13, 'HLL'),
        BasicAction(14, 'HLH'),

        BasicAction(15, 'HHN'),
        BasicAction(16, 'HHL'),
        BasicAction(17, 'HHH'),
    ]
    EMPTY_STATE = BCState(ACTIONS, [0,0,0,0,0,0,0])

    def __init__(self, learning_problem:str, output_color_code:str, rl_agent, pos_reward_sender:ValueSender=None, neg_reward_sender:ValueSender=None, bad_r_thres=0.0):
        from config.settings import MAX_LEN_SIGNALS, IS_SIMULATION

        self.old_valence = float(0.0)
        self.signal_processor=SocialSignalProcessor()

        Environment.__init__(self, learning_problem, output_color_code, rl_agent, pos_reward_sender, neg_reward_sender, bad_r_thres)

        if not IS_SIMULATION:
            ConcreteObserver.__init__(self, [(self.signal_server, SignalEvents.SPOKEN)])

    def get_init_state(self)->State:
        return self.modify_state(None, None)

    def modify_state(self, state: BCState, action: 'Action') -> State:
        return BCEnvironmentBandit.EMPTY_STATE

    def get_reward(self) -> float:
        from config.settings import valence_user_key, arousal_user_key

        val_list=list(self.signal_server.SERVER_QUEUES[valence_user_key])
        arousal_list=list(self.signal_server.SERVER_QUEUES[arousal_user_key])
        explicit_feedback_queue = self.signal_server.SERVER_QUEUES['explicit_feedback']

        # for explicit feedback only, if signal lists are empty
        if len(explicit_feedback_queue)>0:
            reward=explicit_feedback_queue[-1]
            self.send_reward(reward)
            return reward

        new_val, new_arousal=self.signal_processor.mean_valence_arousal(val_list,arousal_list)

        reward=new_val-self.old_valence
        self.old_valence=new_val
        self.send_reward(reward)

        return reward


    def update(self, event):
        if event==SignalEvents.EXPLICIT_REWARD or event==SignalEvents.SPOKEN:
            self.process_rl_step()


