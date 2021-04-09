from seirsplus.models import *
from seirsplus.networks import *
from seirsplus.sim_loops import *
from seirsplus.utilities import *
import networkx
import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt
from seirsplus.FARZ import *

if __name__ == "__main__":

    graph, info = FARZ.generate(farz_params={
                            'n': 100,
                            'm': 3, # mean degree / 2
                            'k': 4, # num communities wased 50 maybe change
                            'alpha': 5.0,                 # clustering param
                            'gamma': -5,                 # assortativity param
                            'beta':  0.9,                 # prob within community edges
                            'r':     1,                  # max num communities node can be part of
                            #'q':     0.1,                 # probability of multi-community membership only if r>1
                            'phi': 10, 'b': 0.0, 'epsilon': 0,
                            'directed': False, 'weighted': False})
    color_map =[]
    community_color= ['g','r','k','b','y']
    for key,value in info.items():
        color_map.append(community_color[value[0]])

    networkx.draw(graph,node_color=color_map)
    plt.show()