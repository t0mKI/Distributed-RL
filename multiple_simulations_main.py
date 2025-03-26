
from config.settings import *
from app.rl.simulation.rl_simulation import SimRLApplication
from app.rl.simulation.logging.cfgs import LoggingCFG, SaveLoggingCFG
from rl.agent import build_agent
from rl.rl_enums import *
from rl.physical_handler import PhysicalAgentSimulation
from rl.environments.simulation.simulated_bc_envs import SimulatedBCEnvironmentBandit, SimulatedBCEnvironment


show_rl_log = False
logging_cfg = LoggingCFG(meta_data_info=True,
                         sar_history=False,
                         snapshot_history=False,
                         mean_rewards=False,
                         percentage_optimal_actions=False,
                         rms=False)
save_logging_cfg = SaveLoggingCFG(sar_history=False,
                                  snapshot_history=False,
                                  mean_rewards=False,
                                  percentage_optimal_actions=False,
                                  rms=False,
                                  output_file='output\\',
                                  safe_plot=True,
                                  fig_path=''
                                  )
#alphas = [0.1, 0.2, 0.3]
alphas = [0.3]
#gammas = [0.5, 0.7, 0.9]
gammas = [0.0]
# strategies = [(SelectionStrategies.EPSILON_GREEDY, [0.05]),
#               (SelectionStrategies.EPSILON_GREEDY, [0.1]),
#               (SelectionStrategies.EPSILON_GREEDY, [0.2])
#               ]
strategies = [(SelectionStrategies.EPSILON_GREEDY, [0.05])
              ]
noise = [0.0, 0.05, 0.1, 0.3]

sim_noise=0.5
SENSOR_NOISE_LB=float(-sim_noise)
SENSOR_NOISE_UB=float(sim_noise)

USER_NOISE_LB=float(-sim_noise)
USER_NOISE_UB=float(sim_noise)
MAX_REWARD=float(1.0)
MIN_REWARD=float(-1.0)

#1000 episodes, 1000 steps== 4 std
#100 episodes, 1000 steps==  15 min
#10 episodes, 1000 steps== min
episodes = 10
steps = 10
preference_change = [5]




def main():
    global show_rl_log, logging_cfg, save_logging_cfg, alphas, gammas, strategies, noise, episodes, steps, preference_change

    # print("bandits")
    # algorithm='bandit'
    #
    # dir = 'plts/' + algorithm
    # for f in os.listdir(dir):
    #     os.remove(os.path.join(dir, f))
    #
    # # Bandits
    # for alpha in alphas:
    #     for gamma in [0.0]:
    #         for strategy in strategies:
    #
    #             begin_prob_experiment = datetime.datetime.now().replace(microsecond=0)
    #             save_logging_cfg.fig_path=algorithm +'/'  + str(QType.STANDARD).replace('.','_') + '_' + 'a' + str(alpha).replace('.','_') + '_g_' + str(gamma).replace('.','_') + '_e' + str(strategy[1][-1]).replace('.','_')
    #
    #             app = SimRLApplication(env_agent_building_fkt=get_bandit,
    #                                    max_episodes=episodes,
    #                                    max_steps=steps,
    #                                    user_preference_change=preference_change,
    #                                    alphas=[alpha],
    #                                    gammas=[gamma],
    #                                    noise_probabilities=noise,
    #                                    strategies=[strategy],
    #                                    seed=SIM_SEED,
    #                                    chart_filter='gamma', #strategy #alpha #gamma
    #                                    eval_type=EvaluationType.REWARD,
    #                                    logging_cfg=logging_cfg,
    #                                    save_logging_cfg=save_logging_cfg
    #                                    )
    #             app.run()
    #
    #             print_duration(time_str='one probability experiment(' + str(len(noise)) + ' probs)', beginning_ts=begin_prob_experiment)


    # print("Q-Learning")
    # algorithm='q_learning'
    # dir = 'plts/' + algorithm
    # for f in os.listdir(dir):
    #     os.remove(os.path.join(dir, f))
    # # q learning
    # for alpha in alphas:
    #     for gamma in gammas:
    #         for strategy in strategies:
    #
    #             begin_prob_experiment = datetime.datetime.now().replace(microsecond=0)
    #             save_logging_cfg.fig_path=algorithm +'/'  + str(QType.STANDARD).replace('.','_') + '_' + 'a' + str(alpha).replace('.','_') + '_g_' + str(gamma).replace('.','_') + '_e' + str(strategy[1][-1]).replace('.','_')
    #
    #             app = SimRLApplication(env_agent_building_fkt=get_q,
    #                                    max_episodes=episodes,
    #                                    max_steps=steps,
    #                                    user_preference_change=preference_change,
    #                                    alphas=[alpha],
    #                                    gammas=[gamma],
    #                                    noise_probabilities=noise,
    #                                    strategies=[strategy],
    #                                    seed=SIM_SEED,
    #                                    chart_filter='gamma', #strategy #alpha #gamma
    #                                    eval_type=EvaluationType.REWARD,
    #                                    logging_cfg=logging_cfg,
    #                                    save_logging_cfg=save_logging_cfg
    #                                    )
    #             app.run()
    #
    #             print_duration(time_str='one probability experiment(' + str(len(noise)) + ' probs)', beginning_ts=begin_prob_experiment)


    # 1000 steps, 1000 episodes== 50 min
    print("Gradient")
    algorithm='gradient'
    dir = 'plts/' + algorithm

    if dir and not os.path.exists(dir):
        os.makedirs(dir)
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    #alphas = [0.9 * BETA, 0.7 * BETA, 0.5 * BETA]
    #alphas=[0.5*BETA]

    # Gradients
    for alpha in alphas:
        for gamma in gammas:
            for strategy in strategies:

                begin_prob_experiment = datetime.datetime.now().replace(microsecond=0)
                save_logging_cfg.fig_path=algorithm +'/'  + str(GradientType.STANDARD).replace('.','_') + '_' + 'a' + str(alpha).replace('.','_') + '_g_' + str(gamma).replace('.','_') + '_e' + str(strategy[1][-1]).replace('.','_')

                app = SimRLApplication(env_agent_building_fkt=get_bandit,
                                       max_episodes=episodes,
                                       max_steps=steps,
                                       user_preference_change=preference_change,
                                       alphas=[alpha],
                                       gammas=[gamma],
                                       noise_probabilities=noise,
                                       strategies=[strategy],
                                       seed=SIM_SEED,
                                       chart_filter='gamma', #strategy #alpha #gamma
                                       eval_type=EvaluationType.REWARD,
                                       logging_cfg=logging_cfg,
                                       save_logging_cfg=save_logging_cfg
                                       )
                app.run()

                print_duration(time_str='one probability experiment(' + str(len(noise)) + ' probs)', beginning_ts=begin_prob_experiment)


##########################################################################
####################       exemplary          ############################
##########################################################################
##########################################################################

def get_bandit():
    env_agent_list = []
    color1 = '\033[30m'

    physical_agent1 = PhysicalAgentSimulation()
    agent_1 = build_agent(algorithm=Algorithm.Q, algorithm_type=QType.STANDARD, selection_strategy=SelectionStrategies.EPSILON_GREEDY,
                          selection_config=float(0.1), env=None, physical_agent=physical_agent1)


    environment_1 = SimulatedBCEnvironmentBandit('Adapting BC Style', color1, agent_1)


    env_agent_list.append((environment_1, agent_1))

    return env_agent_list

def get_q():
    env_agent_list = []
    color1 = '\033[30m'

    physical_agent1 = PhysicalAgentSimulation()
    agent_1 = build_agent(algorithm=Algorithm.Q, algorithm_type=QType.STANDARD, selection_strategy=SelectionStrategies.EPSILON_GREEDY,
                          selection_config=float(0.1), env=None, physical_agent=physical_agent1)

    environment_1 = SimulatedBCEnvironment('Adapting BC Style', color1, agent_1)

    env_agent_list.append((environment_1, agent_1))

    return env_agent_list

def get_grad():
    env_agent_list = []
    color1 = '\033[30m'

    physical_agent1 = PhysicalAgentSimulation()
    agent_1 = build_agent(algorithm=Algorithm.GRADIENT, algorithm_type=GradientType.TDC, selection_strategy=SelectionStrategies.EPSILON_GREEDY,
                          selection_config=float(0.1), env=None, physical_agent=physical_agent1)

    environment_1 = SimulatedBCEnvironment('Adapting BC Style', color1, agent_1)

    env_agent_list.append((environment_1, agent_1))

    return env_agent_list


import datetime
def print_duration( time_str, beginning_ts):
    ending_ts = datetime.datetime.now().replace(microsecond=0)

    duration = ending_ts - beginning_ts
    h, ms = divmod(duration.total_seconds(), 3600)
    min, sec = divmod(ms, 60)

    print('Duration of ' + time_str + ': h: %s, m: %s, s:%s' % (h, min, sec))


##########################################################################
####################       exemplary          ############################
##########################################################################
##########################################################################


if __name__ == '__main__':
    main()