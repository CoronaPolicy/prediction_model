import numpy as np
import networkx as nx
from seirsplus.models import *
from seirsplus.networks import *
import matplotlib.pyplot as plt

plt.style.use('../images/presentation.mplstyle')


def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


def _position_communities(g, partition, **kwargs):
    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos


def _find_between_community_edges(g, partition):
    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        try:
            cj = partition[nj]
        except:
            print("hi")

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges


def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


def main():
    N = 100
    alone_parma = 0.3
    couples_without_kids_param = 0.85
    kids_left_house_p = 0.36
    old_kids = 0.3
    two_kids_young = 0.28
    two_kids_old = 0.032

    three_kids_young = 0.37
    three_kids_old = 0.1

    four_kids_young = 0.37
    four_kids_old = 0.1

    number_of_people_each_household = [1, 4, 4, 2, 2, 3, 3, 4, 5, 8]
    households_data = {'alone': {0.05 * 0.9: [0, 0, alone_parma / 2, alone_parma / 2, 0, 0 * (1 - alone_parma) / 8,
                                              3 * (1 - alone_parma) / 8, 2 * (1 - alone_parma) / 8,
                                              3 * (1 - alone_parma) / 8]},
                       'students_app': {0.05 * 0.1: [0, 0.3, 0.5, 0.2, 0, 0, 0, 0, 0]},
                       'soldier': {0.015: [0.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0]},
                       'couples_without_kids': {
                           0.935 * 0.28 * 0.25: [0, 0, couples_without_kids_param, 1 - couples_without_kids_param,
                                                 0, 0, 0, 0,
                                                 0]},
                       'couples_kids_left_the_house': {0.935 * 0.28 * 0.75: [0, 0, 0, 0, 0, kids_left_house_p,
                                                                             16 * (1 - kids_left_house_p) / 30,
                                                                             10 * (1 - kids_left_house_p) / 30,
                                                                             4 * (1 - kids_left_house_p) / 30]},
                       'couples_with_one_young_kid': {0.935 * 0.18 * 0.9: [0.7, 0.3, 0.5, 0.5, 0, 0, 0, 0, 0]},
                       'couples_with_one_old_kid': {
                           0.935 * 0.18 * 0.1: [0.0, 1.0, 0.0, old_kids, 1 - old_kids, 0, 0, 0, 0]},
                       'couples_with_two_kid': {
                           0.935 * 0.19: [0.5, 0.45, 0.05, two_kids_young, 1 - two_kids_young - two_kids_old,
                                          two_kids_old, 0,
                                          0, 0]},
                       'couples_with_three_kid': {
                           0.935 * 0.17: [0.5, 0.45, 0.05, three_kids_young, 1 - three_kids_old - three_kids_young,
                                          three_kids_old, 0, 0, 0]},
                       'couples_with_four_kid_pluse': {
                           0.935 * 0.18: [0.5, 0.45, 0.05, three_kids_young, 1 - three_kids_old - three_kids_young,
                                          three_kids_old, 0, 0, 0]}}
    layer_info = {'0-9': {'ageBrackets': ['0-9'], 'meanDegree': 8.6, 'meanDegree_CI': (0.0, 17.7)},
                  '10-19': {'ageBrackets': ['10-19'], 'meanDegree': 16.2,
                            'meanDegree_CI': (12.5, 19.8)},
                  '20-39': {'ageBrackets': ['20-29', '30-39'], 'meanDegree': 15.3,
                            'meanDegree_CI': (12.6, 17.9)},
                  '40-59': {'ageBrackets': ['40-49', '50-59'], 'meanDegree': 13.8, 'meanDegree_CI': (11.0, 16.6)},
                  '60+': {'ageBrackets': ['60-69', '70-79', '80+'], 'meanDegree': 13.9,
                          'meanDegree_CI': (7.3, 20.5)}}
    demographic_graphs, individual_ageGroups, households = generate_demographic_contact_network(
        N=N, demographic_data=household_country_data('ISRAEL'),
        distancing_scales=list(), isolation_groups=[], verbose=False,
        layer_info=layer_info, households_data=households_data,
        number_of_people_each_household=number_of_people_each_household)
    g_base = demographic_graphs['baseline']
    node_colors = {k: i for (k, i) in zip(np.unique(individual_ageGroups), range(np.unique(individual_ageGroups).size))}
    colors_list = [node_colors[k] for k in individual_ageGroups]
    situation = {}
    house_size = {}
    house_index = {}
    age = {k: v for (k, v) in zip(range(len(individual_ageGroups)), individual_ageGroups)}
    for i_h, h in enumerate(households):
        indices = h['indices']
        for i in indices:
            situation[i] = h['situation']
            house_size[i] = len(h['indices'])
            house_index[i] = i_h

    nx.set_node_attributes(g_base, situation, 'situation')
    nx.set_node_attributes(g_base, age, 'age')
    nx.set_node_attributes(g_base, house_size, 'house_size')
    nx.set_node_attributes(g_base, house_index, 'house_index')

    optional_situations = np.unique([h['situation'] for h in households])
    optional_situations_index = {k: i for (i, k) in enumerate(optional_situations)}
    ages_indexes = {k: i for (i, k) in enumerate(np.unique(individual_ageGroups))}
    situation_with_indexes = {k: optional_situations_index[v] for k, v in situation.items()}
    age_with_indexes = {k: ages_indexes[v] for k, v in age.items()}

    g = nx.karate_club_graph()
    pos = community_layout(g_base, age_with_indexes)

    nx.draw(g_base, pos, node_color=list(age_with_indexes.values()), node_size=40)

if __name__ == "__main__":
    main()
