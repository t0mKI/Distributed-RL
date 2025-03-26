#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from rl.rl_enums import Algorithm, GradientType
from rl.rl_enums import  SelectionStrategies, QType, BanditType

from utils.util.enum_conversion import StringEnum
import os, sys
from utils.util.math.basis import FourierBasis


'''
Important configuration values for the app
'''
# system variables
MAIN_SCRIPT=os.path.basename(sys.argv[0])   # contains the name of the script, that has been started
FLOAT_MIN=float(0.000000000001)             # value to compare floats for equality

# Path prefixes
ROOT_PATH_PREFIX=Path(".")

# RL configs
ALGORITHM = Algorithm.GRADIENT      # initial algorithm
ALGORITHM_TYPE=GradientType.TDC


SELECTION_STRATEGY=SelectionStrategies.EPSILON_GREEDY

if SELECTION_STRATEGY == SelectionStrategies.EPSILON_GREEDY:
    SELECTION_CONFIG = float(0.1)  # exploration rate
elif SELECTION_STRATEGY == SelectionStrategies.UCB:
    SELECTION_CONFIG = float(0.1)  # ucb value
elif SELECTION_STRATEGY == SelectionStrategies.SOFTMAX:
    SELECTION_CONFIG = float(0.1)  # softmax temperature tau


# state-based specific cfg
GAMMA=float(0.9)            # discount factor
DEFAULT_Q_VALUE=float(0.0)  # initial q value
VALUE_ACCURACY=int(10)       # number of decimal places of a q-value


# gradient specific cfg
FOURIER_DIM=int(7)          # dimension of the fourier basis
FOURIER_ORDER=int(1)
NUM_ACTIONS=7

BASIS=FourierBasis(dimension=FOURIER_DIM, order=FOURIER_ORDER, allowed_coupling=1)
BETA= float(1.0/ (BASIS.dimension() ) )
ALPHA=0.9*BETA
LAMBDA=1.0        # lambda


LOG_FILENAME = "log.json"
RL_PATH="data/rl_data"

# Simulation configs
EPISODES_NUM = int(10)                             # max number of episodes
STEPS_NUM = int(1000)#equals 10 updates           # max number of agent steps
NOISE_PROBABILITES = [0.0,0.05, 0.1, 0.3]        # value for simulating user/sensor noise
EPISODE_COUNTER = int(0)                           # initial value for the episode counter
STEP_COUNTER = int(1)                              # initial value for the step counter
STATE_REMAINING_NUMBER=int(5)

IS_SIMULATION = True       # indicating whether program is simulating
SIM_SEED = 0
MAX_REWARD=float(1.0)
MIN_REWARD=float(-1.0)
PREFERENCES_COUNTER=1
USER_PREFERENCE_CHANGE=[]#500] # int(15),int(26) # step, at which the user preferences change for the second time
BONUS_REWARD=float(0.5)     # bonus reward, when last reward was lower than the current

sim_noise=0.5
SENSOR_NOISE_LB=float(-sim_noise)
SENSOR_NOISE_UB=float(sim_noise)

USER_NOISE_LB=float(-sim_noise)
USER_NOISE_UB=float(sim_noise)

SAVE_POLICY = False
LOAD_POLICY = False
SAVE_GRAPHS = True
SHOW_GRAPHS = True
SAVE_LOG= True
SHOW_RL_LOG= not  IS_SIMULATION

# plot parameters
from utils.util.plot import Linestyle
# line_styles=[
#     Linestyle('#4D4D4D', '--', ' ', 2.5),
#     Linestyle('#60BD68', '--', ' ', 2.5),
#     Linestyle('#5DA5DA', '--', ' ', 1.5),
#     Linestyle('#F15854', '--', ' ', 1.5),
# ]

line_styles=[
    Linestyle('#4D4D4D', '--', ' ', 2.5),
    Linestyle('navy', '--', ' ', 2.5),
    Linestyle('darkorange', '--', ' ', 1.5),
    Linestyle('forestgreen', '--', ' ', 1.5),
]

FIG_SIZE_X=16     #5
FIG_SIZE_Y= 8   #4


# bc parameter
from socket import SocketKind
from utils.app.parser import BCParser
BC_PARSER=BCParser()


# tcp socket parameter
IP='127.0.0.1'
PORT=50000
BUFFERED_CONNECTION_TRIALS=1
DATA_SIZE=1024

from network.socket_communication.signal_server import SignalServer
#network
MAX_LEN_SIGNALS=1000

valence_user_key = 'valence_user'
arousal_user_key = 'arousal_user'
valence_agent_key = 'valence_agent'
arousal_agent_key = 'arousal_agent'
answer_agent_key='answer_agent'
user_turn_wav='user_turn_wav'

rl_action_key='rl_action_key'
pos_r_key='pos_r_key'
neg_r_key='neg_r_key'


QUEUE_CONFIG={valence_user_key: [0.0, MAX_LEN_SIGNALS],
              arousal_user_key: [0.0, MAX_LEN_SIGNALS],
              valence_agent_key: [0.0, MAX_LEN_SIGNALS],
              arousal_agent_key: [0.0, MAX_LEN_SIGNALS],
              #answer_agent_key:[None, 1],
              #'speaking': [0.0 , MAX_LEN_SIGNALS],
              'explicit_feedback': [None, 1],
              user_turn_wav:[float(0.0), MAX_LEN_SIGNALS]
                                           }

SIGNALS=(
         #('speaking', '127.0.0.1', 9999, SocketKind.SOCK_DGRAM, {'callback_fkt':SignalServer.handle_speaking}),
         ('explicit_feedback', '127.0.0.1', 1234, SocketKind.SOCK_DGRAM, {'callback_fkt':SignalServer.handle_explicit_feedback}),
        (valence_agent_key, '127.0.0.1', 5000, SocketKind.SOCK_DGRAM, {'callback_fkt': SignalServer.handle_valence_agent}),
        (arousal_agent_key, '127.0.0.1', 5001, SocketKind.SOCK_DGRAM, {'callback_fkt': SignalServer.handle_arousal_agent}),
        (valence_user_key, '127.0.0.1', 5002, SocketKind.SOCK_DGRAM, {'callback_fkt': SignalServer.handle_valence_user}),
        (arousal_user_key, '127.0.0.1', 5003, SocketKind.SOCK_DGRAM, {'callback_fkt': SignalServer.handle_arousal_user}),
        (user_turn_wav, '127.0.0.1', 5004, SocketKind.SOCK_DGRAM, {'callback_fkt': SignalServer.handle_user_turn_wav})
         # (user_turn_wav, '127.0.0.1', 5004, SocketKind.SOCK_STREAM, {'callback_fkt':SignalServer.handle_user_turn_wav,
         #                                                             'xml_translator': TurnXMLTranslator(),
         #                                                             'protocol': ChunkedFramingProtocol(),
         #                                                             'distributor': TurnDistributor(),
         #                                                             'buffer_size': 4096,
         #                                                             'shutdown_socket': False, 'timeout': 120.0})
         #('xml', '127.0.0.1', 1111, SocketKind.SOCK_DGRAM, {'callback_fkt':SignalServer.handle_xml})
         )

SPEAKING_THRESHOLD=0.5
NEG_R_THRESHOLD=0.5

MAX_TIME_BC=2.0
MAX_TIME_LAST_TRIGGER=1.0
ACTION_SENDING_RATE=10

from network.socket_communication.reward_client import ValueSender
SENDERS={
valence_agent_key:ValueSender(('127.0.0.1', 6000, 4096, SocketKind.SOCK_DGRAM)),
arousal_agent_key:ValueSender(('127.0.0.1', 6001, 4096, SocketKind.SOCK_DGRAM)),
valence_user_key:ValueSender(('127.0.0.1', 6002, 4096, SocketKind.SOCK_DGRAM)),
arousal_user_key:ValueSender(('127.0.0.1', 6003, 4096, SocketKind.SOCK_DGRAM)),

rl_action_key:ValueSender(('127.0.0.1', 6004, 4096, SocketKind.SOCK_DGRAM)),
pos_r_key:ValueSender(('127.0.0.1', 6005, 4096, SocketKind.SOCK_DGRAM)),
neg_r_key:ValueSender(('127.0.0.1', 6006, 4096, SocketKind.SOCK_DGRAM))
}


class Output(StringEnum):
    Q_VALUE_PRE = '------------------Q-Values------------------'
    Q_VALUE_SUF = '--------------------------------------------'
    NEW_EPISODE = '------------------New Episode------------------'
    NEW_EXPERIMENT = '------------------New Experiment------------------'
    NEW_EXPERIMENT_SUF = '--------------------------------------------------'

    GRAPH_STR = 'Noise Probability'
    U_NOISE='u_noise'
    S_NOISE='s_noise'