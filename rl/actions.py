import abc


class Action():
    '''
    Basic class for an Action of a learning :py:class:`rl.agent.Agent`
    '''

    #__metaclass__ = abc.ABCMeta

    def __init__(self, hash_id: int):
        self.hash_id = hash_id

    @abc.abstractmethod
    def __hash__(self):
        """
        Must be implemented and return a single hash value of current action instance
        :return:
        """
        pass

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def copy(self):
        """
        Creates full copy of the action instance
        :return:
        """
        pass

    @abc.abstractmethod
    def apply(self, *args):
        """
        Applies action to current state information.

        :param _state: State
            The current state information (optional)
        :param instance: PhysicalHandler
            The execution of the action can be passed to this component
        :return:
        """
        pass


##########################################################################################################

class BasicAction(Action):
    '''
    This class extends :class:`Action` with :py:class:`rl.rl_enums.ActionNames`
    '''

    def __init__(self, id:int, name:str):
        self.name=name
        Action.__init__(self, id)

    def __str__(self):
        return self.name

    def copy(self):
        return BasicAction(self.hash_id, self.name)

    from rl.state import State
    from rl.physical_handler import PhysicalAgent
    def apply(self, _state: State, instance: PhysicalAgent):
        instance.apply(state=_state,
                       action=self
                       )

    def __eq__(self, other):
        return super(BasicAction, self).__eq__(other) and \
               self.name==other.name

    def __hash__(self):
        """
        Must be implemented and return a single hash value of current action instance
        :return:
        """
        return self.hash_id


class BCAction(BasicAction):
    '''
    This class extends :class:`Action` with :py:class:`rl.rl_enums.ActionNames`
    '''

    def __init__(self, id:int, name:str):
        BasicAction.__init__(self, id, name)

    def copy(self):
        return BCAction(self.hash_id, self.name)

    def __eq__(self, other):
        return super(BasicAction, self).__eq__(other)

    def __hash__(self):
        """
        Must be implemented and return a single hash value of current action instance
        :return:
        """
        return self.hash_id

##########################################################################################################