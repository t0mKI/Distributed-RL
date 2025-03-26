from rl.state import State
from rl.actions import Action


class AR():
    '''
    This class is for logging one rl step
    '''

    def __init__(self, action:Action,reward:float):
        self.action=action
        self.reward=reward

    def get_action(self):
        return self.action
    def get_reward(self):
        return self.reward

    def __eq__(self, other):
        return self.action.__eq__(other.action)

    def __str__(self):
        return str(self.action) + ' ' + str(self.reward)


class SAR(AR):
    '''
    This class extends :py:class:`AR` with states
    '''

    def __init__(self, state:State, action:Action,reward:float):
        super().__init__(action, reward)
        self.state=state

    def get_state(self):
        return self.state

    def __eq__(self, other):
        return self.state.__eq__(other.state) and super().__eq__(other)
        #and\ self._reward == other._reward

    def __str__(self):
        return str(self.state) + ' ' + super().__str__()


class QSnapshot():
    def __init__(self, q_real, q_approximated):
        self.q_real=q_real
        self.q_approximated=q_approximated
