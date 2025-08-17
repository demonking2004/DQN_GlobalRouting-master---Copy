# route_driver.py
# TF2/Keras runner for the global routing DQN — no Router class, no TF1 session, no circular imports.

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

import Initializer as init
import GridGraph as graph
import TwoPinRouterASearch as twoPinASearch
import MST as tree

import tensorflow as tf
import numpy as np
import sys, argparse, operator, math, os, random
import DQN_Implementation  # must NOT import this file back
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required by mpl for 3D)

# ---- Reproducibility
SEED = 10701
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

def _safe_makedirs(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _remove_same_grid_pairs(pairs):
    """Remove pin pairs whose (x,y,z) coords are identical."""
    cleaned = []
    for a, b in pairs:
        if a[:3] != b[:3]:
            cleaned.append([a, b])
    return cleaned

def _two_pin_list_for_net(gridParameters, netNum):
    """Build two-pin list via MST for a single net (dedup pins on same grid)."""
    netPinList, netPinCoord = [], set()
    for j in range(0, gridParameters['netInfo'][netNum]['numPins']):
        pin = (
            int((gridParameters['netInfo'][netNum][str(j+1)][0]-gridParameters['Origin'][0])/gridParameters['tileWidth']),
            int((gridParameters['netInfo'][netNum][str(j+1)][1]-gridParameters['Origin'][1])/gridParameters['tileHeight']),
            int(gridParameters['netInfo'][netNum][str(j+1)][2]),
            int(gridParameters['netInfo'][netNum][str(j+1)][0]),
            int(gridParameters['netInfo'][netNum][str(j+1)][1])
        )
        if pin[0:3] in netPinCoord:
            continue
        netPinList.append(pin)
        netPinCoord.add(pin[0:3])

    base_pairs = []
    for k in range(len(netPinList)-1):
        base_pairs.append([netPinList[k], netPinList[k+1]])

    # MST decomposition
    mst_pairs = tree.generateMST(base_pairs)
    mst_pairs = _remove_same_grid_pairs(mst_pairs)
    return mst_pairs

def DRL_implementation(filename, globali):
    # ---------- Read grid + build base search graph
    grid_info = init.read(filename)
    gridParameters = init.gridParameters(grid_info)

    gridgraph_env = graph.GridGraph(gridParameters)
    capacity = gridgraph_env.generate_capacity()
    _ = gridgraph_env.generate_grid()

    gridGraphSearch = twoPinASearch.AStarSearchGraph(gridParameters, capacity)

    # ---------- Sort nets by half wire length (desc)
    halfWireLength = init.VisualGraph(gridParameters).bounding_length()
    sortedHalfWireLength = sorted(halfWireLength.items(), key=operator.itemgetter(1), reverse=True)
    netSort = [int(idx) for idx, _ in sortedHalfWireLength]

    # ---------- Build two-pin lists (MST) for all nets
    twopinListCombo = []
    twopinListComboCleared = []
    for netNum in range(len(gridParameters['netInfo'])):
        mst_pairs = _two_pin_list_for_net(gridParameters, netNum)
        twopinListCombo.append(mst_pairs)

        # "Cleared" version uses the original vanilla pairs filtered by same-cell pins
        # The original code intended to keep the vanilla order; we approximate it by MST result here,
        # but still remove identical-grid pairs for safety.
        twopinListComboCleared.append(mst_pairs[:])

    twoPinEachNetClear = [len(pairs) for pairs in twopinListComboCleared]

    # ---------- Get A* routes for burn-in targets (per-net)
    routeListMerged = []
    routeListNotMerged = []

    for idx in range(len(gridParameters['netInfo'])):
        netNum = int(sortedHalfWireLength[idx][0])
        twoPinList = _two_pin_list_for_net(gridParameters, netNum)

        # Run A* for all two-pin pairs of this net
        routeListSingleNet = []
        for twoPinPair in twoPinList:
            pinStart, pinEnd = twoPinPair
            route, cost = twoPinASearch.AStarSearchRouter(pinStart, pinEnd, gridGraphSearch)
            routeListSingleNet.append(route)

        # Merge unique locations preserving order
        merged = []
        for path in routeListSingleNet:
            for loc in path:
                if loc not in merged:
                    merged.append(loc)

        routeListMerged.append(merged)
        routeListNotMerged.append(routeListSingleNet)

    # ---------- Flatten two-pin lists (no net grouping) for env
    twopinlist_nonet = [pair for net in twopinListCombo for pair in net]

    # ---------- Configure environment (GridGraph) for DQN
    gridgraph_env.max_step = 100
    gridgraph_env.twopin_combo = twopinlist_nonet
    gridgraph_env.net_pair = [len(net_pairs) for net_pairs in twopinListCombo]

    # ---------- GPU memory growth (optional)
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass

    # ---------- DQN Agent (TF2, no session)
    model_path = '../model/'
    data_path = '../data/'
    environment_name = 'grid'
    _safe_makedirs(model_path)
    _safe_makedirs(data_path)

    agent = DQN_Implementation.DQN_Agent(environment_name, None, gridgraph_env)

    # ---------- Burn-in memory from A* demonstrations
    graphcaseBurnIn = graph.GridGraph(gridParameters)
    graphcaseBurnIn.max_step = 10000

    observationCombo = []
    actionCombo = []
    rewardCombo = []
    observation_nextCombo = []
    is_terminalCombo = []

    for _rep in range(300):
        for i_net in range(gridParameters['numNet']):
            if not routeListMerged[i_net]:
                continue
            goal = routeListMerged[i_net][-1]
            graphcaseBurnIn.goal_state = (goal[3], goal[4], goal[2], goal[0], goal[1])
            for j in range(len(routeListMerged[i_net]) - 1):
                position = routeListMerged[i_net][j]
                nextposition = routeListMerged[i_net][j+1]
                graphcaseBurnIn.current_state = (position[3], position[4],
                                                 position[2], position[0], position[1])

                observationCombo.append(graphcaseBurnIn.state2obsv())

                action = graph.get_action(position, nextposition)
                actionCombo.append(action)

                graphcaseBurnIn.step(action)
                rewardCombo.append(graphcaseBurnIn.instantreward)
                observation_nextCombo.append(graphcaseBurnIn.state2obsv())
                is_terminalCombo.append(False)

            if is_terminalCombo:
                is_terminalCombo[-1] = True

    # reset replay memory before burn-in (fresh training run)
    agent.replay = DQN_Implementation.Replay_Memory()
    agent.burn_in_memory_search(
        observationCombo, actionCombo, rewardCombo,
        observation_nextCombo, is_terminalCombo
    )

    # ---------- Reinitialize gridgraph inside agent before train
    agent.gridParameters = gridParameters
    agent.gridgraph.max_step = 100
    agent.goal_state = None
    agent.init_state = None
    agent.gridgraph.capacity = capacity
    agent.gridgraph.route = []
    agent.gridgraph.twopin_combo = twopinlist_nonet
    agent.gridgraph.twopin_pt = 0
    agent.gridgraph.twopin_rdn = None
    agent.gridgraph.reward = 0.0
    agent.gridgraph.instantreward = 0.0
    agent.gridgraph.best_reward = 0.0
    agent.gridgraph.best_route = []
    agent.gridgraph.route_combo = []
    agent.gridgraph.net_pair = twoPinEachNetClear
    agent.gridgraph.instantrewardcombo = []
    agent.gridgraph.net_ind = 0
    agent.gridgraph.pair_ind = 0
    agent.gridgraph.posTwoPinNum = 0
    agent.gridgraph.passby = np.zeros_like(capacity)
    agent.previous_action = -1

    # ---------- Train
    savepath = model_path
    episodes = agent.max_episodes

    solution_combo_filled, reward_plot_combo, reward_plot_combo_pure, solutionTwoPin, posTwoPinNum = \
        agent.train(len(gridgraph_env.twopin_combo), twoPinEachNetClear, netSort, savepath, model_file=None)

    # ---------- Post-processing / outputs
    _safe_makedirs('solutionsDRL')

    # Build a flat list of two-pin endpoints for optional plotting dots
    twoPinListPlotRavel = []
    for pair in twopinlist_nonet:
        twoPinListPlotRavel.append(pair[0])
        twoPinListPlotRavel.append(pair[1])

    if posTwoPinNum >= len(gridgraph_env.twopin_combo):
        # Reward plots
        n = np.linspace(1, episodes, len(reward_plot_combo))
        plt.figure()
        plt.plot(n, reward_plot_combo)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.savefig(f'test_benchmark_{globali+1}.DRLRewardPlot.jpg')
        plt.close()

        n = np.linspace(1, episodes, len(reward_plot_combo_pure))
        plt.figure()
        plt.plot(n, reward_plot_combo_pure)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.savefig(f'test_benchmark_{globali+1}.DRLRewardPlotPure.jpg')
        plt.close()

        # Save raw rewards
        filenameplot = f'{filename}.rewardData'
        np.save(filenameplot, reward_plot_combo)

        # Dump routed solution
        out_path = f'solutionsDRL/test_benchmark_{globali+1}.gr.DRLsolution'
        with open(out_path, 'w+') as f:
            twoPinSolutionPointer = 0
            routeListMerged_out = solution_combo_filled

            for i_net in range(gridParameters['numNet']):
                singleNetRouteCache = set()
                indicator = i_net  # maintain original order for output header
                netName = gridParameters['netInfo'][indicator]['netName']
                netID = gridParameters['netInfo'][indicator]['netID']
                f.write(f'{netName} {netID} 0\n')

                for j in range(len(routeListMerged_out[indicator])):
                    one_path = routeListMerged_out[indicator][j]
                    for k in range(len(one_path)-1):
                        a = one_path[k]
                        b = one_path[k+1]

                        edge_ab = (a[3], a[4], a[2], b[3], b[4], b[2])
                        edge_ba = (b[3], b[4], b[2], a[3], a[4], a[2])

                        if edge_ab in singleNetRouteCache:
                            continue
                        singleNetRouteCache.add(edge_ab)
                        singleNetRouteCache.add(edge_ba)

                        diff = [abs(a[2]-b[2]), abs(a[3]-b[3]), abs(a[4]-b[4])]
                        if diff[1] > 2 or diff[2] > 2:
                            continue
                        if diff[1] == 2 or diff[2] == 2:
                            continue
                        if diff[0] == 0 and diff[1] == 0 and diff[2] == 0:
                            continue
                        if diff[0] + diff[1] + diff[2] >= 2:
                            continue

                        f.write(f'({int(a[0])},{int(a[1])},{a[2]})-({int(b[0])},{int(b[1])},{b[2]})\n')

                    twoPinSolutionPointer += 1
                f.write('!\n')

        # 3D plot of RL solution
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(0.75, 2.25)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        x_meshP = np.linspace(0, gridParameters['gridSize'][0]-1, 200)
        y_meshP = np.linspace(0, gridParameters['gridSize'][1]-1, 200)
        x_mesh, y_mesh = np.meshgrid(x_meshP, y_meshP)
        z_mesh = np.ones_like(x_mesh)
        ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.3)
        ax.plot_surface(x_mesh, y_mesh, 2*z_mesh, alpha=0.3)
        plt.axis('off')

        for twoPinRoute in solutionTwoPin:
            xs, ys, zs = [], [], []
            for node in twoPinRoute:
                xs.append(node[3]); ys.append(node[4]); zs.append(node[2])
            ax.plot(xs, ys, zs, linewidth=2.5)

        plt.xlim([0, gridParameters['gridSize'][0]-1])
        plt.ylim([0, gridParameters['gridSize'][1]-1])
        plt.savefig(f'DRLRoutingVisualize_test_benchmark_{globali+1}.png')
        plt.close()

        # 2D plot by layer color
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        for routeList in routeListNotMerged:
            for path in routeList:
                for k in range(len(path)-1):
                    pair_x = [path[k][3], path[k+1][3]]
                    pair_y = [path[k][4], path[k+1][4]]
                    pair_z = [path[k][2], path[k+1][2]]
                    if pair_z[0] == pair_z[1] == 1:
                        ax2.plot(pair_x, pair_y, linewidth=2.5)
                    if pair_z[0] == pair_z[1] == 2:
                        ax2.plot(pair_x, pair_y, linewidth=2.5)
        ax2.axis('scaled')
        ax2.invert_yaxis()
        plt.xlim([-0.1, gridParameters['gridSize'][0]-0.9])
        plt.ylim([-0.1, gridParameters['gridSize'][1]-0.9])
        plt.axis('off')
        plt.savefig(f'DRLRoutingVisualize_test_benchmark2d_{globali+1}.png')
        plt.close()
    else:
        print("DRL did not complete all two-pin pairs with current max episodes.")

def main():
    _safe_makedirs('solutionsDRL')
    reduced_path = 'benchmark_reduced'
    _safe_makedirs(reduced_path)

    files = sorted([f for f in os.listdir(reduced_path) if f.endswith('.gr')])
    if not files:
        print("No benchmarks found in 'benchmark_reduced'.")
        return

    for i, fname in enumerate(files):
        filename = os.path.join(reduced_path, fname)
        print('******************************')
        print(f'Working on {filename}')
        DRL_implementation(filename, i)

if __name__ == "__main__":
    main()

plt.tight_layout()
