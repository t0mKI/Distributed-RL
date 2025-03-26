
class LoggingCFG():
    def __init__(self,meta_data_info=False, sar_history=False, snapshot_history=False, mean_rewards=False, percentage_optimal_actions=False, rms=False):
        self.meta_data_info=meta_data_info
        self.sar_history=sar_history
        self.snapshot_history=snapshot_history
        self.mean_rewards=mean_rewards
        self.percentage_optimal_actions=percentage_optimal_actions
        self.rms=rms

class SaveLoggingCFG(LoggingCFG):
    def __init__(self, sar_history=False, snapshot_history=False, mean_rewards=False, percentage_optimal_actions=False, rms=False, output_file='.', safe_plot=True, fig_path:str=''):
        LoggingCFG.__init__(self, sar_history, snapshot_history, mean_rewards, percentage_optimal_actions, rms)
        self.output_file=output_file
        self.safe_plot=safe_plot
        self.fig_path=fig_path