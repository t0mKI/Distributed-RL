import abc


class PhysicalAgent():
    '''
    Directly implements the real app of an action. (gui, socket_handler)

    Basic class for the physical agent.
    '''


    def __init__(self, parser:'Parser'):
        '''

        :param parser:
            The parser is meant to translate the action to a real action
        '''
        self.action_parser=parser

    @abc.abstractmethod
    def apply(self, **kwargs):
        """
        Needs to implement the execution logic for the used physical instance
        :param args: ArgsType
            Data to be sent to physical handler
        :return:
        """
        pass

############################################################################################################

from socket import SocketKind
from network.socket_communication.action_client import ActionSender

class PhysicalAgentCommunicator(PhysicalAgent):
    '''
    This class extends :class:`ArtificialFrame` and implements a physical simulation
    '''
    from utils.app.parser import Parser
    def __init__(self, client_cfg: (str, int, int, SocketKind), parser: Parser):
        PhysicalAgent.__init__(self, parser=parser)
        self.action_sender = ActionSender(client_cfg)


    def apply(self, **kwargs):
        state = ((kwargs['state'])).copy()
        action=((kwargs['action'])).copy()
        parsed_action =self.action_parser.__apply__(state=state, action=action)
        self.action_sender.send_action(parsed_action)

############################################################################################################


class PhysicalAgentSimulation(PhysicalAgent):
    '''
    This class extends :class:`ArtificialFrame` and implements a physical simulation
    '''

    def __init__(self):
        pass

    def apply(self, **kwargs):
        pass



from network.socket_communication.action_client import ActionSender
class PhysicalAgentPasser(PhysicalAgent):
    '''
    This class extends :class:`ArtificialFrame` and implements a physical simulation
    '''
    from utils.app.parser import Parser
    def __init__(self, parser: Parser, signal_server_setter, algorithm: 'Algorithm'):
        PhysicalAgent.__init__(self, parser=parser)
        self.signal_server_setter=signal_server_setter
        self.algorithm=algorithm


    def apply(self, **kwargs):
        from rl.rl_enums import Algorithm

        action = ((kwargs['action'])).copy()

        if self.algorithm==Algorithm.BANDIT:
            parsed_action =self.action_parser.__apply__(action=action)
        else:
            state = ((kwargs['state'])).copy()
            parsed_action = self.action_parser.__apply__(state=state, action=action)
        self.signal_server_setter(parsed_action)

