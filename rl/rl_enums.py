from utils.util.enum_conversion import IntEnum, StringEnum
from enum import auto


class EvaluationType(StringEnum):
    REWARD='Reward'
    RMS='RMS'
    OPTIMAL_ACTION='Optimal Action'
    AVG_REWARDS_EPISODE='Avg Rewards Episode'
    AVG_REWARDS_CUMULATIVE = 'Avg Rewards Cumulative'
    AVG_REWARDS_DEVIATED = 'Avg Rewards Deviated'

    def __init__(self, _:str):
      pass

    def __eq__(self, other):
        return self.value == other.value

class SimulationValues(StringEnum):
    def __init__(self, _:str, letter:str, configs:[]):
        self.letter=letter
        self.configs=configs

    def __eq__(self, other):
        return self.value == other.value and\
            self.letter==other.letter and\
            self.configs==other.configs


class SelectionStrategies(SimulationValues):
    '''
    This Enum defines states for emotions. Each state has at least one :py:class:`ActionNames`
    '''

    # epsilon=[float(0.1),
    #          float(0.15),
    #          float(0.2),
    #          float(0.25)]
    #
    # confidence = [float(0.1),
    #               float(0.15),
    #               float(0.2),
    #               float(0.25)]
    #
    # tau=[float(0.1),
    #      float(0.15),
    #      float(0.2),
    #      float(0.25)]

    EPSILON_GREEDY = "epsilon" , r'$ \epsilon $', [0.05,0.1,0.15,0.2]
    UCB = "ucb" , r'c', []
    SOFTMAX = "softmax", r'$ \tau $', []

    def __init__(self, _: str, letter:str, configs:[]):
        super().__init__(_, letter, configs)


class LearningValues(SimulationValues):
    ALPHA='Alpha', r'$ \alpha $', [0.1,0.2,0.4,0.5] # str(r'$ \alpha $')
    GAMMA='Gamma', r'$ \gamma $',[]                # str(r'$ \gamma $')

    def __init__(self, _: str, letter:str, configs:[]):
        super().__init__(_,letter, configs)


class AlgorithmInformation(IntEnum):
    def __init__(self, _:int, name:str):
        self.info_name=name

    def __eq__(self, other):
        return self.value == other.value and\
        self.info_name==other.info_name


class Algorithm(AlgorithmInformation):
    BANDIT= 0, 'K-Armed-Bandit'
    Q = 1, 'Q-Learning'
    GRADIENT = 2, 'Gradient'
    RANDOM = 3, 'Random'
    NONE = 4, 'None'

    def __init__(self, _: int, name:str):
        super().__init__(_, name)



class AlgorithmType(AlgorithmInformation):
    def __init__(self, _:int, name:str):
        super().__init__(_,name)

class GradientType(AlgorithmType):
    STANDARD = auto(), 'Standard'
    TDC = auto(), 'TDC'
    GTD2 = auto(), 'GTD2'
    ACC_TRACE = auto(), 'Acc_Trace'
    REP_TRACE = auto(), 'Rep_Trace'

    def __init__(self, _:int, name:str):
        super().__init__(_, name)

class QType(AlgorithmType):
    STANDARD=auto(), 'Standard'
    SARSA= auto(), 'Sarsa'

    def __init__(self, _:int, name:str):
        super().__init__(_, name)

class BanditType(AlgorithmType):
    STANDARD= auto(), 'Standard'

    def __init__(self, _:int, name:str):
        super().__init__(_, name)


