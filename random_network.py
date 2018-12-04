#!/usr/bin/env python3
import networkx as nx
import matplotlib.pyplot as plt
import yaml
import argparse
import collections
import numpy as np
from scipy import stats
import time
import os
import itertools
import seaborn as sns

import logger as my_logger

logger = my_logger.get_logger('random_network')
TS = str(time.time())
PLOTS_FOLDER = "plots/"


def readconf(file):
    """
    """
    with open(file, 'r') as ymlfile:
        conf = yaml.load(ymlfile)
    return conf


def generate_erdos_reneye_network(n=100, k=4.0, save=False):
    """
    """
    # Create disconnected network
    ernetwork = nx.Graph()
    ernetwork.add_nodes_from([i for i in range(n)])
    iteration = 0
    avgdegree = 0.0
    degreesum = 0
    while avgdegree <= k:
        logger.debug(
            "Iteration = %i, avg degree = %f" % (iteration, avgdegree)
        )
        # pick randomly a node
        first_node = np.random.choice(list(ernetwork.nodes()))
        # pick randomply another node
        second_node = np.random.choice(list(ernetwork.nodes()))
        # Make a link between nodes
        if first_node != second_node \
                and first_node not in ernetwork.neighbors(second_node):
            ernetwork.add_edge(first_node, second_node)
            degreesum += 2
        else:
            logger.debug("Here is the same or already connected")
        # Compute average degree
        avgdegree = degreesum/n
        # Update iterative variables
        iteration += 1
    # Draw network
    logger.debug(ernetwork.degree())
    if save:
        nx.draw(ernetwork, with_labels=True, font_weight='bold')
        plt.savefig(PLOTS_FOLDER + "ernetwork_graph_" + TS + ".png")

    return ernetwork


def generate_barabasi_albert_network(n=100, edges=4, save=False):
    """
    """
    # Creating the fully connected inital graph
    banetwork = nx.complete_graph(edges)
    # Add up to N nodes
    current_node = banetwork.number_of_nodes()
    degreesum = sum(d[1] for d in banetwork.degree())
    while current_node < n:
        # Add a new node
        banetwork.add_node(current_node)
        # Connect the new node
        prob = [d[1] / degreesum for d in banetwork.degree()]
        neighbors = np.random.choice(
            list(banetwork.nodes()), p=prob, size=4, replace=False
        )
        for neighbor in neighbors:
            banetwork.add_edge(current_node, neighbor)
            degreesum += 2
        # Updating counter variables
        current_node += 1
    if save:
        nx.draw(banetwork, with_labels=True, font_weight='bold')
        plt.savefig(PLOTS_FOLDER + "banetwork_graph_" + TS + ".png")

    return banetwork


def plot_degree_dist(network, save=False):
    """
    """
    degree_sequence = sorted([d for n, d in network.degree()], reverse=True)
    data = np.array(degree_sequence)
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())

    # Statistics
    mean = r"$\mu=" + str("{0:.2f}".format(data.mean())) + "$"
    std = r"$\sigma=" + str("{0:.2f}".format(data.std())) + "$"

    sns.set_context("paper")
    sns.set_style("whitegrid")

    fig = sns.scatterplot(deg, cnt)
    # Adding rug plot
    sns.rugplot(degree_sequence, height=0.01, **{"color": "skyblue"})

    # Set the Title of the graph from here
    fig.axes.set_title('Degree distribution ' + mean + ', ' + std, size=12)
    # Set the xlabel of the graph from here
    fig.set_xlabel("Degree")
    # Set the ylabel of the graph from here
    fig.set_ylabel("Frequency")
    # Set the ticklabel size and color of the graph from here
    fig.tick_params(labelsize=14, labelcolor="black")
    # Set legends
    fig.legend(labels=['Degree distrubution', 'Rug plot'])

    plt.show()


def plot_degree_dist_log_log(network):
    """
    """
    degree_sequence = sorted([d for n, d in network.degree()], reverse=True)
    data = collections.Counter(degree_sequence)
    deg, cnt = zip(*data.items())

    sns.set_context("paper")
    sns.set_style("whitegrid")
    # Initialize figure and ax
    fig, ax = plt.subplots()

    # Set the scale of the x-and y-axes
    ax.set(xscale="log", yscale="log")

    fig = sns.scatterplot(deg, cnt,  ax=ax)
    # Adding rug plot
    sns.rugplot(degree_sequence, height=0.01, **{"color": "skyblue"})
    # Set the Title of the graph from here
    fig.axes.set_title('Degree distribution ', size=12)
    # Set the xlabel of the graph from here
    fig.set_xlabel("Degree (log)")
    # Set the ylabel of the graph from here
    fig.set_ylabel("Frequency (log)")
    # Set the ticklabel size and color of the graph from here
    fig.tick_params(labelsize=14, labelcolor="black")
    # Set legends
    fig.legend(labels=['Degree distrubution', 'Rug plot'])

    plt.show()


def fit_normal_curve(network):
    """
    """
    degree_sequence = sorted([d for n, d in network.degree()], reverse=True)
    data = np.array(degree_sequence)
    # Statistics
    mean = r"$\mu=" + str("{0:.2f}".format(data.mean())) + "$"
    std = r"$\sigma=" + str("{0:.2f}".format(data.std())) + "$"

    sns.set_context("paper")
    sns.set_style("whitegrid")
    fig = sns.distplot(
        degree_sequence, fit=stats.norm, kde=False, fit_kws={'color': 'red'}
    )

    # Set the Title of the graph from here
    fig.axes.set_title('Degree distribution', size=12)
    # Set the xlabel of the graph from here
    fig.set_xlabel("Degree")
    # Set the ylabel of the graph from here
    fig.set_ylabel("Frequency")
    # Set the ticklabel size and color of the graph from here
    fig.tick_params(labelsize=14, labelcolor="black")
    # Set legends
    fig.legend(
        labels=['Normal curve fitted, '+mean + ' '+std, 'Degree distrubution']
    )

    plt.show()


def fit_exp_curve(network):
    """
    """
    degree_sequence = sorted([d for n, d in network.degree()], reverse=True)

    # Statistics
    loc, scale = stats.expon.fit(degree_sequence)
    logger.info("loc = %f, scale = %f" % (loc, scale))
    lam = r"$\lambda=" + str("{0:.2f}".format(1.0/scale)) + "$"

    sns.set_context("paper")
    sns.set_style("whitegrid")
    fig = sns.distplot(
        degree_sequence, fit=stats.expon, kde=False, fit_kws={'color': 'red'}
    )

    # Set the Title of the graph from here
    fig.axes.set_title('Degree distribution ', size=12)
    # Set the xlabel of the graph from here
    fig.set_xlabel("Degree")
    # Set the ylabel of the graph from here
    fig.set_ylabel("Frequency")
    # Set the ticklabel size and color of the graph from here
    fig.tick_params(labelsize=14, labelcolor="black")
    # Set legends
    fig.legend(
        labels=['Exponential curve fitted ' + lam, 'Degree distribution'])

    plt.show()


def fit_least_square_curve(network):
    """
    """
    degree_sequence = sorted([d for n, d in network.degree()], reverse=True)
    data = collections.Counter(degree_sequence)
    deg, cnt = zip(*data.items())
    deg = np.array(deg)

    sns.set_context("paper")
    sns.set_style("whitegrid")
    # Initialize figure and ax
    fig, ax = plt.subplots()

    # Set the scale of the x-and y-axes
    ax.set(xscale="log", yscale="log")

    # Least square
    A = np.vstack([deg, np.ones(len(deg))]).T
    m, c = np.linalg.lstsq(A, cnt, rcond=None)[0]
    logger.info("m = %f, c = %f" % (m, c))

    fig = sns.scatterplot(deg, cnt, ax=ax)

    plt.plot(deg, m*deg + c, 'r')

    # remove the top and right line in graph
    # sns.despine()
    # Set the Title of the graph from here
    fig.axes.set_title('Degree distribution ', size=12)
    # Set the xlabel of the graph from here
    fig.set_xlabel("Degree (log)")
    # Set the ylabel of the graph from here
    fig.set_ylabel("Frequency (log)")
    # Set the ticklabel size and color of the graph from here
    fig.tick_params(labelsize=14, labelcolor="black")
    # Set legends
    fig.legend(labels=['Least square curve fitted', 'Degree distrubution'])

    plt.show()


def plot_cumulative_dist(network):
    """
    """
    degree_sequence = sorted([d for n, d in network.degree()], reverse=True)
    data = collections.Counter(degree_sequence)
    deg, cnt = zip(*data.items())
    deg = np.array(deg)

    # Cumultaive dist
    nodesum = len(network.nodes())
    cum = [c/nodesum for c in cnt]
    cum = list(itertools.accumulate(cum))

    sns.set_context("paper")
    sns.set_style("whitegrid")

    # Least square
    A = np.vstack([deg, np.ones(len(deg))]).T
    m, c = np.linalg.lstsq(A, cum, rcond=None)[0]
    logger.info("m = %f, c = %f" % (m, c))

    # Initialize figure and ax
    fig, ax = plt.subplots()

    # Set the scale of the x-and y-axes
    ax.set(xscale="log", yscale="log")

    fig = sns.scatterplot(deg, cum, ax=ax)

    plt.plot(deg, m*deg + c, 'r')

    # remove the top and right line in graph
    # sns.despine()
    # Set the Title of the graph from here
    fig.axes.set_title('Cumulative Degree distribution ', size=12)
    # Set the xlabel of the graph from here
    fig.set_xlabel("Degree (log)")
    # Set the ylabel of the graph from here
    fig.set_ylabel("Frequency (log)")
    # Set the ticklabel size and color of the graph from here
    fig.tick_params(labelsize=14, labelcolor="black")
    # Set legends
    fig.legend(labels=['Least square curve fitted', 'Degree distrubution'])

    plt.show()


def initialize():
    """
    """
    # Creating plots folder if it doesn't exist
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    # Reading conf and args
    ap = argparse.ArgumentParser(description='Generate networks')
    ap.add_argument(
        "-c", "--config", required=False, help="Configuration file (yml)",
        nargs='?', const=str, type=str, default="config.yml", dest="config"
    )
    ap.add_argument(
        "-t", "--test", required=False, help="Test run",
        nargs='?', const=bool, type=bool, default=False, dest="test"
    )
    ap.add_argument(
        "-s", "--save", required=False, help="Save all the possible plots",
        nargs='?', const=bool, type=bool, default=False, dest="save"
    )
    ap.add_argument(
        "-n", "--network", required=False, help="Which network to generate",
        nargs='?', const=str, type=str, default="all", dest="network"
    )
    args = ap.parse_args()
    logger.debug("arguments %s" % (str(args)))
    conf = readconf(args.config)

    return args, conf


def main():
    """
    """
    args, conf = initialize()
    if "ernetwork" == args.network or "all" == args.network:
        ernconf = conf['erdos_renye_net']
        np.random.seed(ernconf['randseed'])
        logger.info(
            "The algorithm is running with the following configuration: %s" % (
                ernconf
            )
        )
        ernetwork = \
            generate_erdos_reneye_network() if args.test else \
            generate_erdos_reneye_network(ernconf['N'], ernconf['K'])
        plot_degree_dist(ernetwork)
        fit_normal_curve(ernetwork)
    if "banetwork" == args.network or "all" == args.network:
        banconf = conf['barabasi_albert_net']
        np.random.seed(banconf['randseed'])
        logger.info(
            "The algorithm is running with the following configuration: %s" % (
                banconf
            )
        )
        banetwork = \
            generate_barabasi_albert_network() if args.test else \
            generate_barabasi_albert_network(banconf['N'], banconf['edges'])
        plot_degree_dist(banetwork)
        fit_exp_curve(banetwork)
        plot_degree_dist_log_log(banetwork)
        fit_least_square_curve(banetwork)
        plot_cumulative_dist(banetwork)


if __name__ == "__main__":
    main()
