from collections import deque
from network.socket.general.multiplexing_servers import MultiplexServer
from collections.abc import Callable
import struct
import math

from utils.sw_patterns.behavioral.observer import ConcreteObservable
from utils.util.enum_conversion import StringEnum
from socket import SocketKind


class SignalEvents(StringEnum):
    SPOKEN='spoken'
    EXPLICIT_REWARD='explicit_reward'

    def __init__(self, _:str):
      pass

    def __eq__(self, other):
        return self.value == other.value


import threading
class SignalServer(MultiplexServer, ConcreteObservable):

    LISTENING = False
    instance=None
    mutex = threading.RLock()
    counter = 0
    dflow_communicator=None
    dm=None


    def __init__(self, queue_config:{str: [int, int]}, *signals:[(str,str,int,SocketKind, Callable)], ):
        '''
        :param *queue_config:{name: [init_val, max_len]}
        :param signals: [(str,str,int,Callable[[Arg1Type, Arg2Type], ReturnType])]
                each signal contains a 'name', 'ip' , port and callback funktion for a function handler (see UdpFunctionHandler)
        '''
        from config.settings import SENDERS

        if SignalServer.instance is None:
            SignalServer.instance=self
            MultiplexServer.__init__(self, *signals)

            signal_events = [s for s in SignalEvents]
            ConcreteObservable.__init__(self, signal_events)

            self.start_time_bc=-math.inf
            self.bc_s=[(float(0.0), 0)]
            self.senders=SENDERS

            self.SERVER_QUEUES = {}
            for name, config in queue_config.items():
                self.SERVER_QUEUES[name] = deque([config[0]] if config[0] is not None else [], maxlen=config[1])
                print('Server-queue "' + name + '" configured')


    @staticmethod
    def get_instance(queue_config: {str: [int, int]}, *signals: [(str, str, int, SocketKind, Callable)]):

        if SignalServer.instance is None:
            SignalServer(queue_config, *signals)
        return SignalServer.instance

############################################################################################
############################################################################################
############################################################################################
    #METHODS FOR HANDLING VALENCE/ AROUSAL FROM DIFFERENT PORTS
    @staticmethod
    def handle_valence_user(datagram):

        from config.settings import QUEUE_CONFIG, SIGNALS, valence_user_key
        signal_server = SignalServer.get_instance(QUEUE_CONFIG, *SIGNALS)

        #extract and save value
        signal_server.mutex.acquire()
        raw_valence = datagram[4:8]
        valence_val = struct.unpack('f', raw_valence)[0]
        signal_server.SERVER_QUEUES[valence_user_key].append(valence_val)
        signal_server.mutex.release()

    @staticmethod
    def handle_arousal_user(datagram):
        from config.settings import QUEUE_CONFIG, SIGNALS, arousal_user_key
        signal_server = SignalServer.get_instance(QUEUE_CONFIG, *SIGNALS)

        #extract and save value
        signal_server.mutex.acquire()
        raw_arousal = datagram[4:8]
        arousal_val = struct.unpack('f', raw_arousal)[0]
        signal_server.SERVER_QUEUES[arousal_user_key].append(arousal_val)
        signal_server.mutex.release()

    @staticmethod
    def handle_valence_agent(datagram):
        from config.settings import QUEUE_CONFIG, SIGNALS, valence_agent_key
        signal_server = SignalServer.get_instance(QUEUE_CONFIG, *SIGNALS)

        # extract and save value
        signal_server.mutex.acquire()
        raw_valence = datagram[4:8]
        valence_val = struct.unpack('f', raw_valence)[0]
        signal_server.SERVER_QUEUES[valence_agent_key].append(valence_val)
        signal_server.mutex.release()

    @staticmethod
    def handle_arousal_agent(datagram):
        from config.settings import QUEUE_CONFIG, SIGNALS, arousal_agent_key
        signal_server = SignalServer.get_instance(QUEUE_CONFIG, *SIGNALS)

        # extract and save value
        signal_server.mutex.acquire()
        raw_arousal = datagram[4:8]
        arousal_val = struct.unpack('f', raw_arousal)[0]
        signal_server.SERVER_QUEUES[arousal_agent_key].append(arousal_val)
        signal_server.mutex.release()


    @staticmethod
    def handle_user_turn_wav(datagram):
    # def handle_user_turn_wav(audio_turns:[AudioTurn]):
        from config.settings import QUEUE_CONFIG, SIGNALS, user_turn_wav, MAX_TIME_LAST_TRIGGER
        import time
        signal_server = SignalServer.get_instance(QUEUE_CONFIG, *SIGNALS)

        # extract and save value
        signal_server.mutex.acquire()
        raw_speaking_timestamp = datagram[4:8]
        speaking_timestamp_val = struct.unpack('f', raw_speaking_timestamp)[0]

        # if last trigger signal is passed long enough
        if speaking_timestamp_val-signal_server.SERVER_QUEUES[user_turn_wav][-1] > MAX_TIME_LAST_TRIGGER*1000:
            signal_server.SERVER_QUEUES[user_turn_wav].append(speaking_timestamp_val)

            # in this case the bc from initialization is already in the queue, no RL step has to be triggered
            if len(signal_server.bc_s)==2:
                signal_server.bc_s.append(signal_server.bc_s[-1])   # duplicate to not ever enter this condition
            # trigger RL step
            else:
                signal_server.notify_observers(SignalEvents.SPOKEN)
            signal_server.start_time_bc = time.time()

        signal_server.mutex.release()
############################################################################################
############################################################################################
############################################################################################
    '''periodic function to call'''
    @staticmethod
    def sending_action():
        from config.settings import QUEUE_CONFIG, SIGNALS, MAX_TIME_BC,valence_agent_key,arousal_agent_key, valence_user_key, arousal_user_key,rl_action_key
        import time
        signal_server = SignalServer.get_instance(QUEUE_CONFIG, *SIGNALS)

        signal_server.mutex.acquire()
        end_time=time.time()

        # time for bc is not appropriate anymore
        # send no bc application
        if (end_time-signal_server.start_time_bc)>=MAX_TIME_BC:
            agent_action_idx, agent_expr = signal_server.bc_s[0]


        # time for bc is appropriate!!!!!!!!!
        else:
            # send bc from RL
            agent_action_idx, agent_expr = signal_server.bc_s[-1]
            print("appropriate bc " + str(end_time - signal_server.start_time_bc) + "seconds")


        agent_action_str = struct.pack('f', agent_action_idx)
        signal_server.senders[rl_action_key].send_msg(agent_action_str)

        #################################################################
        # in every case alter emotions
        valence_agent=struct.pack('f',signal_server.SERVER_QUEUES[valence_agent_key][-1])
        arousal_agent=struct.pack('f',signal_server.SERVER_QUEUES[arousal_agent_key][-1])

        if agent_expr==0:
            pass
        elif agent_expr==1:
            pass
        elif agent_expr==2:
            pass

        #send agent emotions
        signal_server.senders[valence_agent_key].send_msg(valence_agent)
        signal_server.senders[arousal_agent_key].send_msg(arousal_agent)

        #################################################################

        #send user emotions
        valence_user=struct.pack('f',signal_server.SERVER_QUEUES[valence_user_key][-1])
        arousal_user=struct.pack('f',signal_server.SERVER_QUEUES[arousal_user_key][-1])
        signal_server.senders[valence_user_key].send_msg(valence_user)
        signal_server.senders[arousal_user_key].send_msg(arousal_user)

        # send withouth bc to agent
        signal_server.mutex.release()



    @staticmethod
    def handle_explicit_feedback(raw_reward):
        from config.settings import QUEUE_CONFIG, SIGNALS
        signal_server = SignalServer.get_instance(QUEUE_CONFIG, *SIGNALS)

        reward_val = struct.unpack('d', raw_reward)[0]
        reward_queue=signal_server.SERVER_QUEUES['explicit_feedback']
        reward_queue.append(reward_val)
        signal_server.notify_observers(SignalEvents.EXPLICIT_REWARD)

    @staticmethod
    def handle_answer_agent(raw_str):
        from config.settings import QUEUE_CONFIG, SIGNALS
        signal_server = SignalServer.get_instance(QUEUE_CONFIG, *SIGNALS)

        signal_server.mutex.acquire()
        text_of_agent = raw_str.decode()
        text_of_agent = text_of_agent.rstrip('\x00')
        answer_queue=signal_server.SERVER_QUEUES['answer_agent']
        answer_queue.append(text_of_agent)

        signal_server.notify_observers(SignalEvents.SPOKEN)
        signal_server.clear_all_queues()
        signal_server.mutex.release()


    @staticmethod
    def handle_speaking(raw_speaking):
        from config.settings import QUEUE_CONFIG, SIGNALS, SPEAKING_THRESHOLD
        signal_server = SignalServer.get_instance(QUEUE_CONFIG, *SIGNALS)
        signal_server.mutex.acquire()

        speaking_val = struct.unpack('f', raw_speaking)[0]
        speaking_queue = signal_server.SERVER_QUEUES['speaking']
        old_val = speaking_queue[-1]

        # new speaking turn and signal collecting is beginning
        if old_val < SPEAKING_THRESHOLD and speaking_val >= SPEAKING_THRESHOLD:
            #print("User begins to speak - old: " + str(old_val)+ "new: " + str(speaking_val))

            signal_server.LISTENING=True
            #print("signal true")

        #
        elif old_val >= SPEAKING_THRESHOLD and speaking_val < SPEAKING_THRESHOLD:
            signal_server.LISTENING = False

        speaking_queue.append(speaking_val)
        signal_server.mutex.release()


    @staticmethod
    def clear_all_queues():
        from config.settings import QUEUE_CONFIG, SIGNALS
        signal_server = SignalServer.get_instance(QUEUE_CONFIG, *SIGNALS)

        for name, config in QUEUE_CONFIG.items():
            if name != 'speaking':
                signal_server.SERVER_QUEUES[name] = deque([config[0]] if config[0] is not None else [], maxlen=config[1])



    @staticmethod
    def add_bc(bc_for_agent):
        from config.settings import QUEUE_CONFIG, SIGNALS, user_turn_wav
        signal_server = SignalServer.get_instance(QUEUE_CONFIG, *SIGNALS)

        signal_server.bc_s.append(bc_for_agent)
