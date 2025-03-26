from config.settings import *
from app.rl.simulation.logging.plotting_data import RLPlottingValues, ExperimentMetaInfo
from rl.rl_enums import SelectionStrategies,EvaluationType
from utils.util.plot import Plot
from config.settings import STEP_COUNTER,STEPS_NUM
from rl.logging.sar import QSnapshot
from app.rl.application import RLApplication
from app.rl.simulation.logging.cfgs import LoggingCFG, SaveLoggingCFG
import datetime



class SimRLApplication(RLApplication):
    '''
    '''

    def __init__(self, env_agent_building_fkt, max_episodes:int, max_steps:int, user_preference_change:[int],
                 alphas:[float], gammas:[float], noise_probabilities:[float],
                 strategies: [(SelectionStrategies,[float])], seed:int, chart_filter:str, eval_type:EvaluationType, logging_cfg: LoggingCFG, save_logging_cfg:SaveLoggingCFG):
        '''

        :param environment:
        :param algorithm:
        :param algorithm_type:
        :param learning_rates:
        :param noise_probabilities:
        :param strategies:
            should be a dictionary, where the key equals the name of an action selection strategy
            e.g. 'epsilon-greedy') and the value is a list of config values for this strategy (e.g. [0.05, 0.1]).
            So one entry in the passed dictionary equals for example: {'epsilon-greedy':[0.05, 0.1]}
        :param seed:
        '''

        RLApplication.__init__(self, env_agent_building_fkt)
        self.environment = self.env_agent_list[0][0]

        self.algorithm=self.environment.rl_agent.algorithm
        self.algorithm_type=self.environment.rl_agent.algorithm_type
        self.user = self.environment.user
        #self.states=self.environment.STATES

        self.max_episodes=max_episodes
        self.max_steps=max_steps
        self.user_preference_change=user_preference_change
        self.alphas=alphas
        self.gammas=gammas
        self.noise_probabilities=noise_probabilities
        self.strategies=strategies
        self.seed=seed
        self.chart_filter=chart_filter
        self.eval_type=eval_type
        self.logging_cfg=logging_cfg
        self.save_logging_cfg=save_logging_cfg
        self.first_time=True
        self.begin_simulation=None
        self.begin_prob_experiment=None
        self.fig_path=save_logging_cfg.fig_path


        self.snapshots_q=None
        self.plot = Plot('Algorithm: ' + str(self.algorithm.info_name) + '\nType: ' + str(self.algorithm_type.info_name)
                         , 'Steps', 'Mean ' + self.eval_type.value , self.save_logging_cfg.safe_plot)


    def run(self):

        self.begin_simulation = datetime.datetime.now().replace(microsecond=0)

        all_experiments=self.run_gammas(self.gammas, self.alphas, self.strategies)
        self.plot.create_fig(list(range(STEP_COUNTER,self.max_steps)),
                             all_experiments, self.eval_type,
                             self.plot.get_prob_line_labels(),
                             line_styles,
                             self.plot.filter_chart_labels(all_experiments, self.chart_filter),
                             [0, self.max_steps, 0, 1.5],# [0, 35, 0, 1.5],
                             [0, self.max_steps/2, self.max_steps],
                             [0, 0.5, 1.0] )


        self.print_duration('overall simulation', self.begin_simulation)

        #self.plot.plot()
        self.plot.save('plts/' + self.fig_path )

    def print_duration(self, time_str, beginning_ts):
        ending_ts = datetime.datetime.now().replace(microsecond=0)

        duration = ending_ts - beginning_ts
        h, ms = divmod(duration.total_seconds(), 3600)
        min, sec = divmod(ms, 60)

        print('Duration of ' + time_str + ': h: %s, m: %s, s:%s' % (h, min, sec))



    def run_gammas(self,  gammas:[float], alphas:[float], strategies: [(SelectionStrategies,[float])]):
        experiment_data = {}
        for gamma in gammas:
            new_data=self.run_learning_rates(gamma, alphas, strategies)
            experiment_data.update(new_data)
        return experiment_data

    def run_learning_rates(self, gamma:float, alphas:[float], strategies: [(SelectionStrategies,[float])] ):
        experiment_data = {}
        for alpha in alphas:
            new_data =self.run_selection_strategies(gamma, alpha, strategies)
            experiment_data.update(new_data)
        return experiment_data

#####################################
# methods for encapsulating the action selection strategy experiments
#####################################

    def run_selection_strategies(self, gamma:float, alpha: float, strategies: [(SelectionStrategies,[float])] ):
        '''
        Iterate over all action selection strategies set in the simulation

        :param alpha:
        :return:
        '''
        experiment_data = {}
        for strategy, configs in strategies:
            new_data=self.run_selection_strategy(gamma, alpha,strategy, configs)
            experiment_data.update(new_data)
        return experiment_data

    def run_selection_strategy(self, gamma:float, alpha:float, strategy:'SelectionStrategy', configs:[float]):
        '''
        Concerning one single strategy this method iterates over every related strategy config value and
        runs for every value all probability experiments
        :param alpha:
        :param strategy:
        :return:
        '''

        experiment_data={}
        for config in configs:
            new_data=self.run_probability_experiments(gamma, alpha, strategy, config)
            experiment_data.update(new_data)
        return experiment_data


#####################################
#methods for encapsulating the probability experiments
#####################################

    def run_probability_experiments(self, gamma:float, alpha: float, strategy: 'SelectionStrategies', strategy_config: float)->[RLPlottingValues]:
        pr_experiments_pdata=[]
        self.environment.rl_agent.alpha = alpha
        self.environment.rl_agent.gamma = gamma
        self.environment.rl_agent.selection_strategy = strategy
        self.environment.rl_agent.selection_config=strategy_config
        self.begin_prob_experiment = datetime.datetime.now().replace(microsecond=0)


        # run a probability experiment for each noiseProbability
        for noise_prob in self.noise_probabilities:

            # add list for adding up all rewards for each step of an experiment
            meta_info=ExperimentMetaInfo(
                                         self.algorithm,
                                         self.algorithm_type,
                                         alpha,
                                         gamma,
                                         strategy,
                                         strategy_config,
                                         user_noise_prob=noise_prob,
                                         sensor_noise_prob=noise_prob)
            experiment_data=self.run_probability_experiment(meta_info)

            experiment_data.print_data(self.logging_cfg)
            experiment_data.save_data(self.save_logging_cfg)

            pr_experiments_pdata.append(experiment_data)

        #end of probability experiment
        #self.print_duration('one probability experiment(' + str(len(self.noise_probabilities)) + ' probs)', self.begin_prob_experiment)

            # alpha+strategy+strategy_val:[plotting_vals_noise_exp1,plotting_vals_noise_exp2, ... ]

        return {meta_info.__str__():pr_experiments_pdata}


    def run_probability_experiment(self,meta_info:ExperimentMetaInfo)->RLPlottingValues:
        import copy
        # filter the very first time, where the env already is initialized correctly by its constructor
        if self.first_time:
            pass
        else:
            self.environment.reset()
        self.user.set_noise_probabilites(meta_info.user_noise_prob,meta_info.sensor_noise_prob)
        self.user.set_seed(self.seed)
        self.snapshots_q=[]

        for episode in range(self.max_episodes):
            self.environment.rl_agent.selection_config = meta_info.strategy_val
            print(str(meta_info.sensor_noise_prob) + " " + str(episode))
            self.run_episode()

        # prepare knowledge of all episodes of the probability experiment
        agent=self.environment.rl_agent
        plotting_values=RLPlottingValues(
                                         meta_info,
                                         copy.deepcopy(agent.avg_rewards_episode),
                                         copy.deepcopy(agent.avg_rewards_cumulative),
                                         copy.deepcopy(agent.avg_rewards_deviated),
                                         copy.deepcopy(agent.sar_history),
                                         self.snapshots_q
                                         )

        return plotting_values



    def run_episode(self):
        # filter the very first time, where the env already is initialized correctly by its constructor
        if self.first_time:
            self.first_time=False
        else:
            self.environment.next_episode()
        self.user.reset_q()  # sets random goal values for reward calculation,
        self.snapshots_q.append([])
        agent=self.environment.rl_agent

        while self.environment.step_counter<self.max_steps:
            if not self.environment.STATES is None:

                # TODO: anytime: remove the dependency on self.environment.STATES, since  some envs could be come with a big state space, which won't be defined in advance
                snapshot=QSnapshot(self.user.get_q(), agent.get_Q(self.environment.STATES))
                try:
                    self.snapshots_q[self.environment.episode_counter-1].append(snapshot)
                except:
                    print('RL_Simulation: snapshot appending failed')

            ######teste decreasing epsilon#############
            # if self.environment.step_counter%20==0 and self.environment.rl_agent.selection_config > 0.0 and not self.environment.rl_agent.selection_config <0.07:
            #     self.environment.rl_agent.selection_config-=0.05
            ######teste decreasing epsilon#############

            self.environment.process_rl_step()

            # change user preferences at different steps
            if self.environment.step_counter in self.user_preference_change:
                self.user.reset_q()

