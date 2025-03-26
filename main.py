import sys
from config.settings import *
from app.rl.simulation.rl_simulation import SimRLApplication
from rl.rl_enums import SelectionStrategies, EvaluationType
from app.rl.application import RLApplication as rl_app
from app.rl.live.rl_live import LiveRLApplication
from app.rl.simulation.logging.cfgs import LoggingCFG, SaveLoggingCFG


def main():
    IS_SIMULATION=False
    if IS_SIMULATION:
        SHOW_RL_LOG=False
        logging_cfg=LoggingCFG(meta_data_info=True,
                                sar_history=False,
                                snapshot_history=False,
                                mean_rewards=False,
                                percentage_optimal_actions=False,
                                rms=False)
        save_logging_cfg=SaveLoggingCFG(sar_history=False,
                                snapshot_history=False,
                                mean_rewards=False,
                                percentage_optimal_actions=False,
                                rms=False,
                                output_file='output\\'
                                )

        app = SimRLApplication(env_agent_building_fkt=rl_app.get_simulated_bc_agent,
                               max_episodes=EPISODES_NUM,
                               max_steps=STEPS_NUM,
                               user_preference_change=USER_PREFERENCE_CHANGE,
                               alphas=[ALPHA],
                               gammas=[GAMMA],
                               noise_probabilities=NOISE_PROBABILITES,
                               strategies=[
                                   (SelectionStrategies.EPSILON_GREEDY,[0.1]),
                               ],
                               seed=SIM_SEED,
                               chart_filter='gamma', #strategy #alpha #gamma
                               eval_type=EvaluationType.REWARD,
                               logging_cfg=logging_cfg,
                               save_logging_cfg=save_logging_cfg
                               )
        app.run()

    else:
        app = LiveRLApplication(env_agent_building_fkt=rl_app.get_live_single_bc_agent)
        app.run()
        import time
        print()
        print('ready to learn')
        print()
        sec=0.0
        i=0
        while True:
            time.sleep(0.1)
            SignalServer.sending_action()

            sec+=0.1
            i+=1
            if i%10==0:
                print(str('action sending alive'))


if __name__ == '__main__':
    main()
