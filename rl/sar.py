from rl.state import State
from rl.actions import Action
from config.settings import FLOAT_MIN, STEPS_NUM, EPISODES_NUM
from utils.util import stdout
import math
from rl.rl_enums import Algorithm
import numpy as np


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

        # return self._state.__eq__(other._state) and\
        # self._action.__eq__(other._action) #\
        # #and\ self._reward == other._reward


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



class ExperimentMetaInfo():
    def __init__(self,  algorithm:'Algorithm', algorithm_type:str, alpha:float, action_selection_strategy:'SelectionStrategy', action_selection_val:float, user_noise_prob:float,sensor_noise_prob:float):
        self.episodes=EPISODES_NUM
        self.steps=STEPS_NUM

        self.algorithm=algorithm
        self.algorithm_type=algorithm_type
        self.alpha=alpha
        self.strategy=action_selection_strategy
        self.strategy_val=action_selection_val
        self.user_noise_prob=user_noise_prob
        self.sensor_noise_prob=sensor_noise_prob

    def __hash__(self):
        return self.__str__().__hash__()

    def __eq__(self, other):
        # print("types")
        # print(type(self) + " " + str(self))
        # print(type(other) + " " + str(other))
        return self.__hash__() == other.__hash__()

    def __str__(self):
        return str(self.alpha) + \
               str(self.strategy) + \
               str(self.strategy_val)

    def __str2__(self):
        return 'Algorithm: ' + str(self.algorithm) + ' of type \'' + str(self.algorithm_type) + '\'\n'


    def print(self):
        from config.settings import Output

        print()
        print(Output.NEW_EXPERIMENT_SUF)
        print(Output.NEW_EXPERIMENT)
        stdout.indent_print(str.format("alpha={0}", self.alpha))
        stdout.indent_print(str.format("strategy={0}", self.strategy.value))
        stdout.indent_print(str.format("strategy_val={0}", self.strategy_val))
        stdout.indent_print(str.format("user_noise={0}", self.user_noise_prob))
        stdout.indent_print(str.format("sensor_noise={0}", self.sensor_noise_prob))
        print(Output.NEW_EXPERIMENT_SUF)
        print(Output.NEW_EXPERIMENT_SUF)

    # def get_experiment_name(self, selection_strategy, user_noise_prob, sensor_noise_prob):
    #     return selection_strategy + selection_value\
    #     Output.U_NOISE + user_noise_prob + '_' + \
    #     Output.S_NOISE + sensor_noise_prob + '_'

class RLPlottingValues():

    def __init__(self, meta_data:ExperimentMetaInfo,
                 avg_rewards_episode:[],avg_rewards_cumulative:[], avg_rewards_deviated:[],
                 sar_history:[], snapshots_q_history:[['QSnapshot']]):
        self.meta_data=meta_data
        self.avg_rewards_episode = avg_rewards_episode
        self.avg_rewards_cumulative = avg_rewards_cumulative
        self.avg_rewards_deviated = avg_rewards_deviated
        self.sar_history=sar_history
        self.snapshots_q_history=snapshots_q_history

        self.mean_rewards=self.mean_sar_history(sar_history)
        self.optimal_actions_perc=self.calc_optimal_actions(snapshots_q_history)
        self.rms = self.calc_rms(snapshots_q_history)

    def calc_rms(self, snapshots_q_history):
        rms_table=[]

        # build a rms table, which contains for each step of each episode one rms value (2d list)
        for episode_index in range(len(snapshots_q_history)):
            rms_table.append([])
            for step_index in range(len(snapshots_q_history[episode_index])):

                q_vals=snapshots_q_history[episode_index][step_index].q_real
                Q_vals=snapshots_q_history[episode_index][step_index].q_approximated
                print('q: '+str(q_vals))
                print('Q: ' + str(Q_vals))
                if not self.meta_data.algorithm==Algorithm.BANDIT:
                    # if algorithm is no bandit, extract the dictionary  {action:val} for the corresponding state
                    sar=self.sar_history[episode_index][step_index]
                    q_vals=q_vals[sar.state]
                    Q_vals=Q_vals[sar.state]

                    print('q: ' + str(q_vals))
                    print('Q: ' + str(Q_vals))

                # calculate rms for every step
                # calculate the sum of all: (error delta)^2
                n=len(q_vals)
                rms_sum=0.0
                for action in q_vals.keys():
                    rms_sum += ( pow(q_vals[action]-Q_vals[action],2) )
                rms= math.sqrt(1/n * rms_sum)
                rms_table[episode_index].append(rms)

        # calc mean of the rms table
        rms_mean=np.array(rms_table).mean(axis=0)

        return rms_mean.tolist()


    def calc_optimal_actions(self, snapshots_q_history):
        optimal_action_chosen=[]

        # build a optimal_action_chosen table, which contains for each step of each episode a 1, if chosen action was optimal
        # 0 otherwise
        for episode_index in range(len(snapshots_q_history)):
            optimal_action_chosen.append([])
            for step_index in range(len(snapshots_q_history[episode_index])):
                q_vals=snapshots_q_history[episode_index][step_index].q_real
                sar = self.sar_history[episode_index][step_index]

                if not self.meta_data.algorithm==Algorithm.BANDIT:
                    # if algorithm is no bandit, extract the dictionary  {action:val} for the corresponding state
                    q_vals=q_vals[sar.state]

                # if one of the optimal actions was chosen-->this step is voted by 1.0, otherwise by 0.0
                optimal_action=0.0
                if sar.action in self.get_optimal_actions(q_vals):
                    optimal_action=1.0

                optimal_action_chosen[episode_index].append(optimal_action)

        # calculate percentage, how often optimal action was chosen
        optimal_action_perc = np.array(optimal_action_chosen).mean(axis=0)
        return optimal_action_perc.tolist()


    def get_optimal_actions(self, goal_actions:{}) -> []:
        '''
        Gets all optimal actions in from the actual q*
        :return:
        '''
        a_max = []
        val_max = -float('inf')
        for action, value in goal_actions.items():
            if value > val_max:
                val_max = value
                a_max = [action]
            elif value == val_max:
                a_max.append(action)
        return a_max

    def print_sar_mean(self):
        output = ''
        step=1
        for step_mean in self.mean_rewards:
            output+=str.format('{0} ({1}) ', step_mean, step)
            step+=1
        print(output)

    def print_snapshots_history(self):
        output=''
        for episode_index in range(len(self.snapshots_q_history)):
            output += str.format("episode{0}:\n", episode_index+1)
            q_real_str = ''
            q_approximated = ''
            for step_index in range(len(self.snapshots_q_history[episode_index])):

                #build q_real str
                q_real_str += str.format("step {0} - q: ", step_index+1)
                for key, val in self.snapshots_q_history[episode_index][step_index].q_real.items():
                    if self.meta_data.algorithm==Algorithm.BANDIT:
                        action=key
                        q_real_str+= str.format("({0}: {1}) ",action, val)
                    else:
                        state=key
                        action_vals=val
                        a_val_str=''
                        for action, v in action_vals.items():
                            a_val_str += str.format(" {0}: {1} ", action, v)

                        q_real_str+=str.format("({0}: {1})   ",state, a_val_str)
                q_real_str += str.format("\n")

                # build q_approximated str
                q_approximated += str.format("step {0} - Q: ", step_index + 1)
                for key, val in self.snapshots_q_history[episode_index][step_index].q_approximated.items():
                    if self.meta_data.algorithm==Algorithm.BANDIT:
                        action = key
                        q_approximated += str.format("({0}: {1}) ", action, val)
                    else:
                        state=key
                        action_vals=val
                        a_val_str=''
                        for action, v in action_vals.items():
                            a_val_str += str.format(" {0}: {1} ", action, v)

                        q_approximated+=str.format("({0}: {1})   ",state, a_val_str)
                q_approximated += str.format("\n")

            output += str.format("{0}\n{1} ", q_real_str, q_approximated)

        print(output)

    def print_sar_history(self):
        output=''
        for episode_index in range(EPISODES_NUM):
            output += str.format("episode{0}:  ", episode_index+1)
            step_str=''
            for step_index in range(len(self.sar_history[episode_index])):
                step_str+= str.format("({0})({1}) ", self.sar_history[episode_index][step_index], step_index+1)
            step_str+='\n'
            output += step_str
        print(output)


    def mean_sar_history(self, sar_history)->[float]:
        mean_rewards=[]

        # build a rms table, which contains for each step of each episode one rms value (2d list)
        for episode_index in range(len(sar_history)):
            mean_rewards.append([])
            for step_index in range(len(sar_history[episode_index])):
                mean_rewards[episode_index].append(sar_history[episode_index][step_index].reward)

        # calc mean of the rewards
        mean = np.array(mean_rewards).mean(axis=0)
        return mean.tolist()


class QSnapshot():
    def __init__(self, q_real, q_approximated):
        self.q_real=q_real
        self.q_approximated=q_approximated

