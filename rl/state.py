import abc
from typing import List
import copy
from utils.util.math.comparison import equalz

##########################################################################################################

class State(abc.ABC):
    '''
    Basic class for a state of a learning :py:class:`scripts.rl.agent.Agent`
    '''

    __metaclass__ = abc.ABCMeta

    def __init__(self, actions: List):
        self.actions = actions

    def __eq__(self, other):
        # action lists have same length
        return self.actions == other.actions

    @abc.abstractmethod
    def __hash__(self):
        """
        Must be implemented and return a single hash value of current state instance
        :return:
        """
        pass

    @abc.abstractmethod
    def copy(self):
        """
        Creates full copy of the state instance
        :return:
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


##########################################################################################################


class VectorConverterState(State):
    '''
    This class extends :class:`State` with entries.

    !!!This kind of state is defined for gradient based rl algorithms!!!
    '''
    def __init__(self, actions: List, entries: []):
        '''

        :param actions:
        :param entries:
            Encodes a state with a list of numbers restricted to 0's and 1's (for convergence).
            !!!This attribute is only needed for gradient based rl-methods!!!
        '''
        super().__init__(actions=actions)
        self.__entries = entries

    def dim(self):
        return len(self.__entries)

    def get_state_vector(self):
        return self.__entries

    def update(self, entries: []):
        self.__entries = entries

    def __eq__(self, other):
        return super().__eq__(other) and \
               self.__entries.__eq__(other.__entries) and \
                self.__hash__() == other.__hash__()

    def get_feature_vector(self):
        return self.get_state_vector()

    def __str__(self):
        return str.format(
            "{0}, {1}",
            self.__hash__(),
            self.__entries
        )



##########################################################################################################

class BCState(VectorConverterState):
    '''
    This class extends :class:`VectorConverterState`
    '''

    def __init__(self, actions: List['Action'], entries=[]):
        super().__init__(actions=actions, entries=entries)

        # binary list to integer conversion
        self.hash_id = int("".join(str(i) for i in entries), 2)

    def __hash__(self):
        return self.hash_id

    def __eq__(self, other):
        return super().__eq__(other)

    def copy(self):
        """
        Return a deep copy
        :return: BehavioralState
        """
        actions_copied = []
        for action in self.actions:
            actions_copied.append(copy.deepcopy(action.copy()))
        state_vector = copy.deepcopy(self.get_state_vector())
        state = BCState(actions=actions_copied,
            entries=state_vector)
        return state

    def __str__(self):
        return str.format(
            "{0}|{1}{2}{3}|{4}{5}{6}",
            self.get_state_vector()[0],
            self.get_state_vector()[1],
            self.get_state_vector()[2],
            self.get_state_vector()[3],
            self.get_state_vector()[4],
            self.get_state_vector()[5],
            self.get_state_vector()[6],
        )
