import abc
from config.settings import *
from rl.agent import build_agent
from rl.rl_enums import *
from rl.physical_handler import PhysicalAgentSimulation
from network.socket_communication.signal_server import SignalServer
from rl.physical_handler import PhysicalAgentPasser
from rl.environments.live.live_bc_envs import BCEnvironment

class RLApplication(abc.ABC):
    '''
    Basic class for a learning agent
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self, env_agent_building_fkt):
        self.env_agent_list=env_agent_building_fkt()

    @abc.abstractmethod
    def run(self):
        pass

    # Live applications
    @staticmethod
    def get_live_single_bc_agent():

        env_agent_list=[]
        color1 = '\033[30m'

        signal_server_setter= SignalServer.add_bc
        pos_r_sender = SENDERS[pos_r_key]
        neg_r_sender = SENDERS[neg_r_key]

        physical_agent1 = PhysicalAgentPasser(BC_PARSER, signal_server_setter, ALGORITHM)
        agent = build_agent(algorithm=ALGORITHM, algorithm_type=ALGORITHM_TYPE, selection_strategy=SELECTION_STRATEGY,
                            selection_config=SELECTION_CONFIG, env=None, physical_agent=physical_agent1)

        environment = BCEnvironment('Personality adaption through bc style', color1, agent)

        env_agent_list.append((environment,agent))

        return env_agent_list


    # Simulated applications
    @staticmethod
    def get_simulated_bc_agent():
        from rl.environments.simulation.simulated_bc_envs import SimulatedBCEnvironment, SimulatedBCEnvironmentBandit
        env_agent_list=[]
        color1 = '\033[30m'

        physical_agent1 = PhysicalAgentSimulation()
        agent_1 = build_agent(algorithm=ALGORITHM, algorithm_type=ALGORITHM_TYPE, selection_strategy=SELECTION_STRATEGY,
                            selection_config=SELECTION_CONFIG, env=None, physical_agent=physical_agent1)

        if ALGORITHM==Algorithm.Q:
            environment_1 = SimulatedBCEnvironmentBandit('Adapting BC Style', color1, agent_1)
        else:
            environment_1 = SimulatedBCEnvironment('Adapting BC Style', color1, agent_1)

        env_agent_list.append((environment_1, agent_1))

        return env_agent_list
