
from rl.environments.live.live_bc_envs import BCEnvironment
from rl.environments.simulation.users.bc_users import BCListener
from config.settings import PREFERENCES_COUNTER

#######################################################################################################################

class SimulatedBCEnvironment(BCEnvironment):
    import copy
    STATES=[
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [0, 1, 0, 0, 1, 0, 0]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [0, 0, 1, 0, 1, 0, 0]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [0, 0, 0, 1, 1, 0, 0]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [0, 1, 0, 0, 0, 1, 0]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [0, 0, 1, 0, 0, 1, 0]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [0, 0, 0, 1, 0, 1, 0]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [0, 1, 0, 0, 0, 0, 1]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [0, 0, 1, 0, 0, 0, 1]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [0, 0, 0, 1, 0, 0, 1]),
        #
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [1, 1, 0, 0, 1, 0, 0]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [1, 0, 1, 0, 1, 0, 0]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [1, 0, 0, 1, 1, 0, 0]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [1, 1, 0, 0, 0, 1, 0]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [1, 0, 1, 0, 0, 1, 0]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [1, 0, 0, 1, 0, 1, 0]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [1, 1, 0, 0, 0, 0, 1]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [1, 0, 1, 0, 0, 0, 1]),
        # BCState(copy.deepcopy(BCEnvironment.ACTIONS), [1, 0, 0, 1, 0, 0, 1])
    ]

    def __init__(self, learning_problem:str, output_color_code:str, rl_agent):
        BCEnvironment.__init__(self, learning_problem, output_color_code, rl_agent)
        self.user = BCListener(self.modify_state, self.get_actions_from_entry, PREFERENCES_COUNTER)


    def get_reward(self) -> float:
        base_reward=self.user.get_base_reward(self.last_state, self.rl_agent.action, self.state)
        reward = self.user.get_realistic_reward(base_reward)
        return reward

    def update(self, event):
        pass


from rl.environments.live.live_bc_envs import BCEnvironmentBandit
from rl.environments.simulation.users.bc_users import BCListenerBandit
class SimulatedBCEnvironmentBandit(BCEnvironmentBandit):
    STATES=[
    ]

    def __init__(self, learning_problem:str, output_color_code:str, rl_agent):
        BCEnvironmentBandit.__init__(self, learning_problem, output_color_code, rl_agent)
        self.user = BCListenerBandit( PREFERENCES_COUNTER)


    def get_reward(self) -> float:
        base_reward=self.user.get_base_reward(self.rl_agent.action)
        reward = base_reward#self.user.get_realistic_reward(base_reward)
        #reward = self.user.get_realistic_reward(base_reward)
        return reward

    def update(self, event):
        pass