from matplotlib import colors as mplcolors
import numpy as np
from matplotlib.cbook import boxplot_stats
import matplotlib.pyplot as plt


# this function is from https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
# (credits to Kerry Halupka)

def get_continuous_cmap(hex_list, float_list=None):
    def hex_to_rgb(value):
        '''
        Converts hex to rgb colours
        value: string of 6 characters representing a hex colour.
        Returns: list length 3 of RGB values'''
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


    def rgb_to_dec(value):
        '''
        Converts rgb to decimal colours (i.e. divides each value by 256)
        value: list (length 3) of RGB values
        Returns: list (length 3) of decimal values'''
        return [v/256 for v in value]
    
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        cdict[col] = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]

    return mplcolors.LinearSegmentedColormap('cmp', segmentdata=cdict, N=256)


def jitter_dots(dots, jitter_by=0.25, along_y=False):
    offsets = dots.get_offsets()
    jittered_offsets = offsets
    # only jitter in the x-direction
    if along_y:
        jittered_offsets[:, 1] += np.random.uniform(-jitter_by,
                                                jitter_by,
                                                offsets.shape[0])
    else:
        jittered_offsets[:, 0] += np.random.uniform(-jitter_by,
                                                    jitter_by,
                                                    offsets.shape[0])
    dots.set_offsets(jittered_offsets)

    
def plot_scatter_and_boxplot(data, position, color = "k", dot_size=1, dot_alpha=0.05,
                             box_linewidth=1.5, box_whisker_linewidth=1):
    
    if len(data) < 200:
        dot_alpha = 2 * dot_alpha
    if len(data) < 100:  # double-dipping intended
        dot_alpha = 2 * dot_alpha
        
    dots = plt.scatter(data, [position] * len(data),
                       color = color, s = dot_size, alpha=dot_alpha)
    jitter_dots(dots, along_y=True)
    
    bps = boxplot_stats(data)[0]
    
    plt.plot([bps["whislo"], bps["whishi"]], [position + 0.4] * 2,
             linewidth=box_whisker_linewidth, color=color, alpha=0.7)
    plt.plot([bps["q1"], bps["q3"]], [position + 0.37] * 2,
             linewidth=box_linewidth, color=color, alpha=0.9, zorder=28)
    plt.plot([bps["q1"], bps["q3"]], [position + 0.43] * 2,
             linewidth=box_linewidth, color=color, alpha=0.9, zorder=29)

    # draw median dot
    plt.scatter(bps["med"], [position + 0.4], s=8, color="k", zorder=30)
    plt.scatter(bps["med"], [position + 0.4], s=3, color="white", zorder=31)