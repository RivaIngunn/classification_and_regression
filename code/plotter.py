import matplotlib.pyplot as plt
import seaborn as sb
class plotter:

    def single_plot_show(self, label_x, label_y, title, save = False, filename = None):
        plt.style.use('seaborn-whitegrid')
        plt.title(title, fontsize='16')
        plt.xlabel(label_x, fontsize='16')
        plt.ylabel(label_y, fontsize='16')
        plt.tick_params(labelsize='12')
        plt.legend(fontsize='12')

        if save:
            plt.savefig('../../visuals/' + filename + '.pdf')

        plt.show()
