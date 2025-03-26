from utils.util import stdout
import abc
import random



class Parser(object):
    """
    Parser class that helps generating behavior by providing translating input
    """
   # __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def __apply__(self, **kwargs):
        pass


##########################################################################################################

from network.socket_communication.signal_server import SignalServer

class BCParser(Parser):
    """
    This class generates behavior based on given :py:class:`rl.actions.Action`
    """

    def __init__(self):
        super().__init__()
        self.action_idx_map={
        'LNN':(float(0.0), 0),
        'LNL':(float(1.0), 1),
        'LNH':(float(2.0), 1),
        'LLN':(float(3.0), 1),
        'LLL':(float(4.0), 2),
        'LLH':(float(5.0), 2),
        'LHN':(float(6.0), 1),
        'LHL':(float(7.0), 2),
        'LHH':(float(8.0), 2),
        'HNN':(float(9.0), 0),
        'HNL':(float(10.0), 1),
        'HNH':(float(11.0), 1),
        'HLN':(float(12.0), 1),
        'HLL':(float(13.0), 2),
        'HLH':(float(14.0), 2),
        'HHN':(float(15.0), 1),
        'HHL':(float(16.0), 2),
        'HHH':(float(17.0), 2),

        }

    def __apply__(self, **kwargs):
        """
        This method translates an action to a extraverted behavior

        :param kwargs: additional parameters
            It contains an action
        :return: filename:str
            File name of a picture that corresponds to the class of the passed action
        """
        action_key = ''


        if 'state' in kwargs:
            state = kwargs['state'].get_state_vector()

            #modify wdh
            if state[0]==0:
                action_key +='L'
            else:
                action_key +='H'

            if state[1]==1:
                action_key +='N'
            elif state[2]==1:
                action_key +='L'
            elif state[3]==1:
                action_key +='H'

            if state[4]==1:
                action_key +='N'
            elif state[5]==1:
                action_key +='L'
            elif state[6]==1:
                action_key +='H'

        else:
            action = kwargs['action']
            action_key=action.name

        action_tuple=self.action_idx_map[action_key]

        #parsed_action will be a tuple of (action_index, expressivity_level)
        return action_tuple