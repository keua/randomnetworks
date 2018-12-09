#!/usr/bin/env python3
import networkx as nx
import matplotlib.pyplot as plt
import yaml
import argparse
import collections
import numpy as np
import time
import os
import itertools
import seaborn as sns
import powerlaw as pwl

import multiprocessing as mp
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pylab import setp
from scipy import stats

import logger as my_logger

logger = my_logger.get_logger('random_network')
TS = str(time.time())
PLOTS_FOLDER = "../plots/"
CONFIG_FOLDER = "../config/"
CONFIG_FILE = CONFIG_FOLDER + "config.yml"
OUTPUT_FOLDER = '../output/'


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
        plt.savefig(PLOTS_FOLDER + TS + "_ernetwork_graph.png")

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
        plt.savefig(PLOTS_FOLDER + TS + "_banetwork_graph.png")

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
    mean = data.mean()
    smean = r"$\mu=" + str("{0:.2f}".format(mean)) + "$"
    std = data.std()
    sstd = r"$\sigma=" + str("{0:.2f}".format(std)) + "$"

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
        labels=['Normal curve fitted, ' + smean + ' ' + sstd,
                'Degree distrubution']
    )

    plt.show()

    return float(mean), float(std)


def fit_exp_curve(network):
    """
    """
    degree_sequence = sorted([d for n, d in network.degree()], reverse=True)

    # Statistics
    loc, scale = stats.expon.fit(degree_sequence)
    logger.info("loc = %f, scale = %f" % (loc, scale))
    lam = 1.0/scale
    strlam = r"$\lambda=" + str("{0:.2f}".format(lam)) + "$"

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
        labels=['Exponential curve fitted ' + strlam, 'Degree distribution'])

    plt.show()

    return float(lam)


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

    return float(m), float(c)


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


def fit_power_law(network):
    """
    """
    panel_label_font = FontProperties().copy()
    panel_label_font.set_weight("bold")
    panel_label_font.set_size(10.0)
    panel_label_font.set_family("sans-serif")

    units = "Barabasi-Albert network"

    fig = plt.figure(figsize=(8, 11))
    degree_sequence = sorted([d for n, d in network.degree()], reverse=True)
    data = np.array(degree_sequence)

    annotate_coord = (0.5, 1.1)
    ax1 = fig.add_subplot(4, 3, 1)
    x, y = pwl.pdf(data, linear_bins=True)
    ind = y > 0
    y = y[ind]
    x = x[:-1]
    x = x[ind]

    # First plot
    ax1.scatter(x, y, color='r', s=.5)
    pwl.plot_pdf(data[data > 0], ax=ax1, color='b', linewidth=2)

    setp(ax1.get_yticklabels(), visible=True)

    ax1.annotate(
        "A", annotate_coord, xycoords="axes fraction",
        fontproperties=panel_label_font
    )
    ax1.set_ylabel(u"p(X)")  # (10^n)")

    ax1in = inset_axes(ax1, width="30%", height="30%", loc=3)
    ax1in.hist(data, density=True, color='b')
    ax1in.set_xticks([])
    ax1in.set_yticks([])

    # Second plot
    ax2 = fig.add_subplot(4, 3, 2, sharex=ax1)
    pwl.plot_pdf(data, ax=ax2, color='b', linewidth=2)
    fit = pwl.Fit(data, xmin=1, discrete=True)
    fit.power_law.plot_pdf(ax=ax2, linestyle=':', color='g')
    p = fit.power_law.pdf()

    ax2.set_xlim(ax1.get_xlim())

    fit = pwl.Fit(data, discrete=True)
    fit.power_law.plot_pdf(ax=ax2, linestyle='--', color='g')

    setp(ax2.get_yticklabels(), visible=False)

    ax2.annotate(
        "B", annotate_coord, xycoords="axes fraction",
        fontproperties=panel_label_font
    )

    ax2.set_xlabel(units)

    # Third plot
    ax3 = fig.add_subplot(4, 3, 3)  # , sharex=ax1)#, sharey=ax2)
    # Fitting maximum likelihood
    fit.power_law.plot_pdf(ax=ax3, linestyle='--', color='g')
    alpha = fit.power_law.alpha
    sigma = fit.power_law.sigma
    logger.info("Alpha = %f, Sigma = %f" % (alpha, sigma))
    fit.exponential.plot_pdf(ax=ax3, linestyle='--', color='r')
    fit.plot_pdf(ax=ax3, color='b', linewidth=2)
    # Comparing
    R, p = fit.distribution_compare('power_law', 'exponential')
    logger.info("R = %f, p = %f" % (R, p))

    setp(ax3.get_yticklabels(), visible=False)

    ax3.set_ylim(ax2.get_ylim())
    ax3.set_xlim(ax1.get_xlim())

    ax3.annotate(
        "C", annotate_coord, xycoords="axes fraction",
        fontproperties=panel_label_font
    )

    plt.show()

    return float(alpha), float(sigma), float(R), float(p)


def play_prisoners_dilemma(net, T=0.05, R=1, P=0, S=-0.1, rounds=10):
    """
    """
    logger.info("Starting playing prisonners dilemma")
    start_time = time.time()

    nx.set_node_attributes(net, 0.0, "payoff")
    nx.set_node_attributes(net, 0, "action")
    rnd = 1
    level_cooperation_per_round = list()
    actions = [0, 1]  # Defect = 0, Cooperate = 1

    def payoff(node, neighbor):
        """ Payoff matrix """
        payoff_mat = [[P, T],
                      [S, R]]
        a1 = net.node[node]['action']
        a2 = net.node[neighbor]['action']
        return payoff_mat[a1][a2]

    def probability_replicate(node, randneighbor):
        """ Compute the probability of replicate action for node """
        wi = net.node[node]['payoff']
        wj = net.node[randneighbor]['payoff']
        kmax = max(net.degree(node), net.degree(randneighbor))
        dmax = max(T, R) - min(S, P)
        return (wj-wi)/(kmax*dmax)

    while rnd <= rounds:
        for node in net.nodes():
            if rnd == 1:
                net.node[node]['action'] = np.random.choice(actions)
            elif len(list(net.neighbors(node))) > 0:
                ranneighbor = np.random.choice(list(net.neighbors(node)))
                p = probability_replicate(node, ranneighbor)
                # Evaluate probability of node
                if np.random.random() < p:
                    net.node[node]['action'] = net.node[ranneighbor]['action']
        # Computing payoff
        for node in net.nodes():
            net.node[node]['payoff'] = 0
            for neighbor in net.neighbors(node):
                node_pay = payoff(node, neighbor)
                net.node[node]['payoff'] += node_pay

        actions = nx.get_node_attributes(net, 'action')
        cooperation = sum(actions.values())
        level_cooperation_per_round.append(cooperation)
        logger.info("Portion of cooperation in round %d: %d" %
                    (rnd, cooperation))
        rnd += 1
        # nx.set_node_attributes(net, 0.0, "payoff")

    logger.info("The simulation took - %s sec -" % (time.time() - start_time))

    return level_cooperation_per_round


def initialize():
    """
    """
    # Creating plots folder if it doesn't exist
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)
    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    # Reading conf and args
    ap = argparse.ArgumentParser(description='Generate networks')
    ap.add_argument(
        "-c", "--config", required=False, help="Configuration file (yml)",
        nargs='?', const=str, type=str, default=CONFIG_FILE, dest="config"
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
    output = dict()

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
        mean, std = fit_normal_curve(ernetwork)
        # Play prisonners dilemma
        simdict = dict()
        rounds = ernconf['ipd']['rounds']
        tempatation = ernconf['ipd']['temptation']
        simulations = ernconf['ipd']['simulations']
        results = []
        pool = mp.Pool(processes=ernconf['ipd']['parallel'])
        for s in range(simulations):
            coopbytemp = dict()
            for t in tempatation:
                results.append(
                    pool.apply_async(
                        play_prisoners_dilemma,
                        (ernetwork,), dict(T=t, rounds=rounds)
                    )
                )
        pool.close()
        pool.join()

        ct = 0
        for s in range(simulations):
            coopbytemp = dict()
            for t in tempatation:
                coopbytemp['cooperation_T='+str(t)] = \
                    {'iteration_' + str(i).zfill(2): int(c)
                    for i, c in enumerate(results[ct].get())}
                ct+=1
            simdict["simulation_"+str(s).zfill(2)] = coopbytemp
            
        # Save to output file
        output["erdos_renye_net"] = dict(
           a_normal_fit=dict(mean=mean, std=std), b_IDP=simdict
        )

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
        lam = fit_exp_curve(banetwork)
        plot_degree_dist_log_log(banetwork)
        c, m = fit_least_square_curve(banetwork)
        plot_cumulative_dist(banetwork)
        alpha, sigma, R, p = fit_power_law(banetwork)
        # Play prisonners dilemma
        simdict = dict()
        rounds = banconf['ipd']['rounds']
        tempatation = banconf['ipd']['temptation']
        simulations = banconf['ipd']['simulations']
        # Specify here as many processes as your processor allows!
        pool = mp.Pool(processes=banconf['ipd']['parallel'])
        results = []
        for s in range(simulations):
            coopbytemp = dict()
            for t in tempatation:
                results.append(
                    pool.apply_async(
                        play_prisoners_dilemma,
                        (banetwork,), dict(T=t, rounds=rounds)
                    )
                )
        pool.close()
        pool.join()

        ct = 0
        for s in range(simulations):
            coopbytemp = dict()
            for t in tempatation:
                coopbytemp['cooperation_T='+str(t)] = \
                    {'iteration_' + str(i).zfill(2): int(c)
                    for i, c in enumerate(results[ct].get())}
                ct+=1
            simdict["simulation_"+str(s).zfill(2)] = coopbytemp

        # Save to output file
        output["barabasi_albert_net"] = dict(
            a_expfit=dict(lambd=lam),
            b_lsquarefit=dict(c=c, m=m),
            c_maxlikelihoodfit=dict(alpha=alpha, sigma=sigma),
            d_mlf_exp_comparison=dict(R=R, p=p),
            f_IPD=simdict
        )

    with open(OUTPUT_FOLDER + TS + '_data.yml', 'w+') as outfile:
        yaml.dump(output, outfile, default_flow_style=False)


if __name__ == "__main__":
    main()
