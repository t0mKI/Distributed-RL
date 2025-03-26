from config.settings import *
import random
from rl.state import VectorConverterState
from rl.actions import Action


# idea: the more similar the bc configuration is to the preference, the more reward:
#stepwise:
    # 0 steps-->reward=1.0
    # 1 steps-->reward=0.5
    # 2 steps-->reward=0.25

class BCListener():
    def __init__(self, state_mod_fct, get_actions_from_entry_fct, preferences_ctr):
        self.rand=random.Random()
        self.last_reward=MIN_REWARD
        self.state_mod_fct=state_mod_fct
        self.get_actions_from_entry=get_actions_from_entry_fct
        self.preferences_ctr=preferences_ctr

        # favourite action for user
        self.preferenced_s_a=[]
        self.preferenced_entries=[]

        self.user_noise_prob=float(0.0)
        self.sensor_noise_prob=float(0.0)

    def get_realistic_reward(self, base_reward):

        # account user noise
        if self.rand.uniform(0, 1) < self.user_noise_prob:
            base_reward += self.rand.uniform(USER_NOISE_LB, USER_NOISE_UB)

        # account sensor noise
        if self.rand.uniform(0, 1) < self.sensor_noise_prob:
            base_reward += self.rand.uniform(SENSOR_NOISE_LB, SENSOR_NOISE_UB)

        reward=min(max(base_reward, MIN_REWARD), MAX_REWARD)
        self.last_reward=reward

        return reward

    def set_noise_probabilites(self, user_noise_prob, sensor_noise_prob):
        self.user_noise_prob=user_noise_prob
        self.sensor_noise_prob=sensor_noise_prob

    def set_seed(self, seed:int):
        self.rand.seed(seed)

    def get_q(self):
        return {}



    def reset_q(self):
        from copy import deepcopy
        from rl.state import BCState
        from rl.environments.simulation.simulated_bc_envs import SimulatedBCEnvironment
        #self.rand.seed(self.seed)

        self.preferenced_s_a=[]
        self.preferenced_entries=[]
        entries = []
        # add preferences_ctr times a preference
        for i in range(self.preferences_ctr):
            new_entry=[]
            while True:
                rep=self.rand.choice([[0], [1]])
                ling_style = self.rand.choice([[1,0,0], [0,1,0], [0,0,1]])
                nod_style = self.rand.choice([[1,0,0], [0,1,0], [0,0,1]])
                new_entry=rep+ling_style+nod_style
                if not new_entry in entries:
                    entries.append(deepcopy(new_entry))
                    actions=self.get_actions_from_entry(new_entry)
                    self.preferenced_s_a.append( (BCState(actions, deepcopy(new_entry)), SimulatedBCEnvironment.ACTIONS[-1]))
                    self.preferenced_entries.append(deepcopy(new_entry))
                    break
        #print()
        pass

    def calc_dist_s_pref(self, state:VectorConverterState, preference_vector:[]):
        s_vector=state.get_state_vector()
        distance=0
        if s_vector[0]!=preference_vector[0]:
            distance+=1

        distance += self.calc_dist_of_vecs(s_vector[1:4], preference_vector[1:4])
        distance += self.calc_dist_of_vecs(s_vector[4:7], preference_vector[4:7])
        return distance

    def calc_dist_of_vecs(self, s_vec:[], pref_vec:[]):
        #index_pos= 100==1 010==2 001==3
        # abs(index_pos_s - index_pos_pref)
        s_pos=s_vec.index(1)+1
        pref_pos=pref_vec.index(1)+1

        return abs(s_pos-pref_pos)



    def get_base_reward(self, last_state:VectorConverterState, action:Action, state:VectorConverterState):
        s_1=last_state
        s_2=state
        a=action
        pref=self.preferenced_entries[-1]
        dist_last_s=self.calc_dist_s_pref(last_state, pref)
        dist_s=self.calc_dist_s_pref(state, pref)
        dist_change=dist_s-dist_last_s

        base_reward=-100000

        # distance is increasing from s to s'
        if dist_change>0:
            base_reward= -1.0

        #distance is decreasing from s to s'
        elif dist_change<0:
            base_reward= 0.5

        # distance remains from s to s'
        else:
            #if s'==preference_state
            pref_s=self.preferenced_s_a[-1][0]
            if state.__eq__(pref_s):
                base_reward= 1.0
            #if s'!=preference_state
            else:
                base_reward= -0.5

        return base_reward



class BCListenerBandit():
    def __init__(self, preferences_ctr):
        self.rand=random.Random()
        self.last_reward=MIN_REWARD
        self.preferences_ctr=preferences_ctr

        # favourite action for user
        self.preferenced_a=[]
        #self.preferenced_entries=[]

        self.user_noise_prob=float(0.0)
        self.sensor_noise_prob=float(0.0)

    def get_realistic_reward(self, base_reward):

        # account user noise
        if self.rand.uniform(0, 1) < self.user_noise_prob:
            base_reward += self.rand.uniform(USER_NOISE_LB, USER_NOISE_UB)

        # account sensor noise
        if self.rand.uniform(0, 1) < self.sensor_noise_prob:
            base_reward += self.rand.uniform(SENSOR_NOISE_LB, SENSOR_NOISE_UB)

        # # account bonus reward
        # if new_reward>=last_reward:
        #     new_reward += BONUS_REWARD

        reward=min(max(base_reward, MIN_REWARD), MAX_REWARD)
        self.last_reward=reward

        return reward

    def set_noise_probabilites(self, user_noise_prob, sensor_noise_prob):
        self.user_noise_prob=user_noise_prob
        self.sensor_noise_prob=sensor_noise_prob

    def set_seed(self, seed:int):
        self.rand.seed(seed)

    def get_q(self):
        return {}


    def reset_q(self):
        from copy import deepcopy
        from rl.state import BCState
        from rl.environments.simulation.simulated_bc_envs import SimulatedBCEnvironmentBandit
        #self.rand.seed(self.seed)

        self.preferenced_a=[]
        # add preferences_ctr times a preference
        for i in range(self.preferences_ctr):
            new_pref_a=deepcopy(self.rand.choice(SimulatedBCEnvironmentBandit.ACTIONS))
            if not new_pref_a in self.preferenced_a:
                self.preferenced_a.append(new_pref_a)


    def get_base_reward(self, action:Action):
        if (action in self.preferenced_a):
            base_reward= 1.0
        else:
            base_reward= -0.5

        return base_reward