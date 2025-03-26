from app.rl.application import RLApplication

class LiveRLApplication(RLApplication):

    def __init__(self, env_agent_building_fkt):
        RLApplication.__init__(self, env_agent_building_fkt)

    def run(self):
        pass

