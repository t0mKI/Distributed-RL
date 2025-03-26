import matplotlib.pyplot as plt
import math
from config.settings import NOISE_PROBABILITES



class Linestyle():
    def __init__(self,color:str, style:str, marker:str, width:float):
        self.color = color
        self.style = style
        self.marker = marker
        self.width = width

class Plot():

    def __init__(self, global_title:str, x_label:str, y_label:str, safe_plt:bool):

        #plt.rcParams.update({'font.size': 24})
        self.global_title = global_title
        self.x_label = x_label
        self.y_label = y_label
        self.fig=None
        self.axes=None
        self.safe_plt=safe_plt


    def save(self, path: str):
        plt.savefig(path + '.png', bbox_inches='tight')
        #plt.savefig(path + '.pdf', bbox_inches='tight')

    def plot(self):
        plt.show()



    # len(multiply_y_values[0]) == len(voI_and_value)
    #line_labels_per_chart==mostly 4
    def create_fig(self, x_values, all_experiments, eval_type, line_labels_per_chart:[str], line_style:[Linestyle], voI_and_value:[str], x_y_min_max:[],x_ticks:[], y_ticks:[]):
        '''
        Preconditions:
        number of chart lines==len(chart_values[0])==len(line_labels_per_chart)==len(line_style)

        :param x_values:
        :param all_experiments:
        :param line_labels_per_chart:[str]
            Contains one legend label for every line of one chart (e.g. 'noise_probability=0.0')
        :param line_style:
        :param voI_and_value:[(str)]
            Contains a list of tuples. Every tuple contains a string for the title and a value, both will be concatenated
            in the title for the corresponding subplot
        :param x_y_min_max:
        :param x_ticks:
        :param y_ticks:
        :param x_label:
        :param y_label:
        :return:
        '''
        from config.settings import FIG_SIZE_X, FIG_SIZE_Y
        from rl.rl_enums import SelectionStrategies,EvaluationType
        pass



        chart_values=[]
        for charts_plotting_vals in all_experiments.values():
            lines=[]
            for plotting_val in charts_plotting_vals:
                if eval_type==EvaluationType.REWARD:
                    lines.append(plotting_val.mean_rewards)
                elif eval_type==EvaluationType.OPTIMAL_ACTION:
                    lines.append(plotting_val.optimal_actions_perc)
                elif eval_type==EvaluationType.RMS:
                    lines.append(plotting_val.rms)

                elif eval_type==EvaluationType.AVG_REWARDS_EPISODE:
                    lines.append(plotting_val.avg_rewards_episode)
                elif eval_type == EvaluationType.AVG_REWARDS_CUMULATIVE:
                    lines.append(plotting_val.avg_rewards_cumulative)
                elif eval_type == EvaluationType.AVG_REWARDS_DEVIATED:
                    lines.append(plotting_val.avg_rewards_deviated)
            chart_values.append(lines)











        # to get approximately a grid with length,width==gridsize
        #grid_size=max(math.ceil(math.sqrt(len(chart_values))),2)

        grid_size =int(math.sqrt(len(chart_values)))

        if grid_size==1:
            fig, ax = plt.subplots(1,1, figsize=(FIG_SIZE_X*grid_size,FIG_SIZE_Y*grid_size))

            for line_index in range(len(chart_values[0])):
                ax.plot(x_values,
                        chart_values[0][line_index],

                        color=line_style[line_index].color,
                        linestyle=line_style[line_index].style,
                        marker=line_style[line_index].marker,
                        linewidth=line_style[line_index].width,
                        label= line_labels_per_chart[line_index])

                # ax_subplot.set_title())
                ax.set(title=voI_and_value[0],
                               xlabel=self.x_label,
                               ylabel=self.y_label)
                ax.axis(x_y_min_max)  # [-335, 35, -334, 64]
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.legend(ncol=2, loc='upper right', title='Noise')

        else:

            self.fig, self.axs = plt.subplots(grid_size, grid_size, figsize=(FIG_SIZE_X*grid_size,FIG_SIZE_Y*grid_size))

            chart_index=0
            for ax_array in self.axs:
                for ax_subplot in ax_array:
                    # for every chart do the following (adding multiple lines and configuring)

                    if chart_index<len(chart_values):
                        for line_index in range(len(chart_values[chart_index])): #iterate over number of lines

                            ax_subplot.plot(x_values,
                                            chart_values[chart_index][line_index],

                                            color=line_style[line_index].color,
                                            linestyle=line_style[line_index].style,
                                            marker=line_style[line_index].marker,
                                            linewidth=line_style[line_index].width,
                                            label= line_labels_per_chart[line_index])

                        #ax_subplot.set_title())
                        ax_subplot.set(title=voI_and_value[chart_index],
                                       xlabel=self.x_label,
                                       ylabel=self.y_label)
                        ax_subplot.axis(x_y_min_max)                       #[-335, 35, -334, 64]
                        ax_subplot.set_xticks(x_ticks)
                        ax_subplot.set_yticks(y_ticks)
                        ax_subplot.legend(ncol=2, loc='lower center', title='Noise')
                        chart_index += 1

            self.fig.suptitle(self.global_title,  fontsize="x-large")

            self.fig.tight_layout(pad=3.08, h_pad=4.0, w_pad=4.0,rect=[0, 0, 1, 0.95])








    def get_prob_line_labels(self):
        line_labels=[]
        for prob in NOISE_PROBABILITES:
            line_labels.append(str(prob*100) + '%')

        return line_labels

    def filter_chart_labels(self, all_experiments:{}, voI_and_value:str):
        chart_labels = []
        for charts_plotting_vals in all_experiments.values():
            meta_data = charts_plotting_vals[0].meta_data
            #label="Experiment with: "
            # if voI_and_value =='strategy':
            #     label+=meta_data.strategy.letter + ' = ' + str(meta_data.strategy_val)
            # elif voI_and_value =='alpha':
            #     label+= str(r'$ \alpha $') + ' = ' + str(meta_data.alpha)
            # elif voI_and_value =='gamma':
            #     label+= str(r'$ \gamma $') + ' = ' + str(meta_data.gamma)

            label= str(r'$ \alpha $') + ' = ' + str(meta_data.alpha) + ', '
            label+= str(r'$ \gamma $') + ' = ' + str(meta_data.gamma) + ', '
            label+=meta_data.strategy.letter + ' = ' + str(meta_data.strategy_val)



            chart_labels.append(label)
        return chart_labels

