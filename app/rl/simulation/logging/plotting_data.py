from config.settings import STEPS_NUM, EPISODES_NUM
import math
from rl.rl_enums import Algorithm
import numpy as np


class ExperimentMetaInfo():
    def __init__(self,  algorithm:'Algorithm', algorithm_type:str, alpha:float, gamma:float, action_selection_strategy:'SelectionStrategy', action_selection_val:float, user_noise_prob:float,sensor_noise_prob:float):
        self.episodes=EPISODES_NUM
        self.steps=STEPS_NUM

        self.algorithm=algorithm
        self.algorithm_type=algorithm_type
        self.alpha=alpha
        self.gamma=gamma
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
        return  str(self.algorithm) + \
                str(self.algorithm_type)  + \
                str(self.alpha) + \
                str(self.gamma) + \
                str(self.strategy) + \
                str(self.strategy_val)

    def __str2__(self):
        return 'Algorithm: ' + str(self.algorithm) + ' of type \'' + str(self.algorithm_type) + '\'\n'

    def __str3__(self):
        from config.settings import Output
        output=''
        output += (Output.NEW_EXPERIMENT_SUF.value + '\n')
        output += (Output.NEW_EXPERIMENT.value + '\n')
        output += (str.format("algorithm={0}", self.algorithm) + '\n')
        output += (str.format("algorithm type={0}", self.algorithm_type) + '\n')
        output += (str.format("alpha={0}", self.alpha) + '\n')
        output += (str.format("gamma={0}", self.gamma) + '\n')
        output += (str.format("strategy={0}", self.strategy.value) + '\n')
        output += (str.format("strategy_val={0}", self.strategy_val) + '\n')
        output += (str.format("user_noise={0}", self.user_noise_prob) + '\n')
        output += (str.format("sensor_noise={0}", self.sensor_noise_prob) + '\n')
        output += (Output.NEW_EXPERIMENT_SUF.value + '\n')
        output += (Output.NEW_EXPERIMENT_SUF.value + '\n')
        return output

    def file_str(self):
        output=''
        output += (str.format("{0}_", self.algorithm) )
        output += (str.format("{0}_", self.algorithm_type) )
        output += (str.format("alpha_{0}_", self.alpha) )
        output += (str.format("gamma{0}_", self.gamma))
        output += (str.format("strategy_{0}_", self.strategy.value))
        output += (str.format("{0}_", self.strategy_val))
        output += (str.format("user_noise_{0}_", self.user_noise_prob))
        output += (str.format("sensor_noise_{0}_", self.sensor_noise_prob))
        return output


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

        self.mean_rewards=self.calc_mean_reward(sar_history)
        #self.optimal_actions_perc=self.calc_percentage_optimal_actions(snapshots_q_history)
        #self.rms = self.calc_rms(snapshots_q_history)

    def count_heat_map_states(self) -> [float]:
        #alle states, die nahe aneinander liegen, werden auch so geplottet
        # Anzahl der Zustände, die besucht wurden
        pass

    def calc_mean_reward(self, sar_history)->[float]:
        mean_rewards=[]

        # build a rms table, which contains for each step of each episode one rms value (2d list)
        for episode_index in range(len(sar_history)):
            mean_rewards.append([])
            for step_index in range(len(sar_history[episode_index])):
                mean_rewards[episode_index].append(sar_history[episode_index][step_index].reward)

        # calc mean of the rewards
        mean = np.array(mean_rewards).mean(axis=0)
        return mean.tolist()

    def calc_rms(self, snapshots_q_history):
        rms_table=[]

        # build a rms table, which contains for each step of each episode one rms value (2d list)
        for episode_index in range(len(snapshots_q_history)):
            rms_table.append([])
            for step_index in range(len(snapshots_q_history[episode_index])):

                q_vals=snapshots_q_history[episode_index][step_index].q_real
                Q_vals=snapshots_q_history[episode_index][step_index].q_approximated
                # print('q: '+str(q_vals))
                # print('Q: ' + str(Q_vals))

                # if algorithm is no bandit, extract the dictionary  {action:val} for the corresponding state
                sar=self.sar_history[episode_index][step_index]
                q_vals=q_vals[sar.state]
                Q_vals=Q_vals[sar.state]

                # print('q: ' + str(q_vals))
                # print('Q: ' + str(Q_vals))

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


    def calc_percentage_optimal_actions(self, snapshots_q_history):
        optimal_action_chosen=[]

        # build a optimal_action_chosen table, which contains for each step of each episode a 1, if chosen action was optimal
        # 0 otherwise
        for episode_index in range(len(snapshots_q_history)):
            optimal_action_chosen.append([])
            for step_index in range(len(snapshots_q_history[episode_index])):

                q_real=snapshots_q_history[episode_index][step_index].q_real
                sar = self.sar_history[episode_index][step_index]

                # all action-values from one state of q_real
                q_s_real=q_real[sar.state]

                # if one of the optimal actions was chosen-->this step is voted by 1.0, otherwise by 0.0
                optimal_action=0.0

                # chosen action -> optimal?
                if sar.action in self.get_optimal_actions(q_s_real):
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


    def get_mean_rewards_str(self):
        output = ''
        output += 'Mean Reward:\n'
        for step in range(len(self.mean_rewards)):
            output += 'step ' + str(step) + ' - ' +  str(self.mean_rewards[step]) + '\n'
        return output

    def get_rms_str(self):
        output = ''
        output += 'RMS:\n'
        for step in range(len(self.mean_rewards)):
            output += 'step ' + str(step) + ' - ' +  str(self.rms[step]) + '\n'
        return output


    def get_percentage_optimal_actions_str(self):
        output = ''
        output += 'Optimal Actions chosen (%):\n'
        for step in range(len(self.optimal_actions_perc)):
            output += ('step ' + str(step) + ' - ' +  str(self.optimal_actions_perc[step]*100) + '%\n')
        return output

    def get_snapshot_history_str(self):
        output = ''
        output += 'Snapshot-History:\n'

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

        return output

    def get_sar_history_str(self):
        output = 'SAR-History:\n'
        for episode_index in range(EPISODES_NUM):
            output += str.format("episode{0}:  ", episode_index+1)
            step_str=''
            for step_index in range(len(self.sar_history[episode_index])):
                step_str+= str.format("({0})({1}) ", self.sar_history[episode_index][step_index], step_index+1)
            step_str+='\n'
            output += step_str
        return output

    def get_meta_data_info(self):
        output=self.meta_data.__str3__()
        return output

    def get_data_str(self, cfg):
        output='\n'

        if cfg.meta_data_info:
            output += (self.get_meta_data_info() + '\n')
        if cfg.sar_history:
            output += (self.get_sar_history_str() + '\n')
        if cfg.snapshot_history:
            output += (self.get_snapshot_history_str() + '\n')
        if cfg.mean_rewards:
            output += (self.get_mean_rewards_str()+ '\n')
        if cfg.percentage_optimal_actions:
            output += (self.get_percentage_optimal_actions_str() + '\n')
        if cfg.rms:
            output += (self.get_rms_str() + '\n')
        return output

    def print_data(self, logging_cfg):
        #print(self.meta_data.__str3__())
        print(self.get_data_str(logging_cfg))

    def save_data(self, save_logging_cfg):
        import os

        data_str=self.get_data_str(save_logging_cfg)

        # Ensure that the directory exists
        output_directory = os.path.dirname( save_logging_cfg.output_file + self.meta_data.file_str() + '.txt')

        # Create the directory if it doesn't exist
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory)

        with open(save_logging_cfg.output_file  + self.meta_data.file_str() + '.txt', "w") as logging_file:
            logging_file.write(data_str)

