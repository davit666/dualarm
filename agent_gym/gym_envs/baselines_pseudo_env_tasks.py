import numpy as np
import time

from itertools import permutations


def baselines_offline_heuristics(cost, mask, parts_mask):
    time0 = time.time()
    cost_node = cost['n']
    cost_node2node = cost['n2n']

    mask_node = mask['n']
    mask_node2node = mask['n2n']

    available_parts = list(np.arange(len(parts_mask)))
    remove_list = []
    for k in range(len(parts_mask)):
        if parts_mask[k] == 0:
            remove_list.append(available_parts[k])
    for rm in remove_list:
        available_parts.remove(rm)
    # print(parts_mask, available_parts)
    ## get best pairs
    transfer_cost, best_pairs = get_best_pairs(cost_node, mask_node, available_parts)
    time1 = time.time()
    print("best pairs:\n", best_pairs)
    print("min transfer cost:\t", transfer_cost)
    print("time used:\t", time1 - time0)

    ## get best order
    curr_pose = (-1, -1)
    transit_cost, best_order = get_best_order(cost_node2node, mask_node2node, best_pairs, curr_pose)
    best_order.reverse()
    time2 = time.time()
    print("best_order:\n", best_order)
    print("min trainsit cost:\t", transit_cost)
    print("time used:\t", time2 - time1)

    ## total cost
    total_cost = transfer_cost + transit_cost

    print(" ")
    print(" ")
    print("total cost:\t", total_cost)
    print("total time used:\t", time2 - time0)

    best_order2 = []
    for p in best_order:
        best_order2.append(p[0])
        best_order2.append(p[1])

    return total_cost, best_order2, time2 - time0


def get_best_pairs(cost_n, mask_n, avai_parts):
    # print("in",avai_parts)
    punish_cost = 1000000
    best_pairs = []
    if len(avai_parts) <= 1:
        # print("out end")
        return 0, best_pairs
    else:
        perm = permutations(avai_parts, 2)
        min_cost = punish_cost
        for p in perm:
            x = p[0]
            y = p[1]
            c = cost_n[x, y]
            m = mask_n[x, y]
            if m == 0:
                continue

            new_avai_parts = avai_parts.copy()
            new_avai_parts.remove(x)
            new_avai_parts.remove(y)

            future_c, pairs = get_best_pairs(cost_n, mask_n, new_avai_parts)
            # print("s", future_c, pairs )
            cost = c + future_c

            if cost < min_cost:
                min_cost = cost
                best_pairs = pairs.copy()
                best_pairs.append(p)
    #             print("update", min_cost,best_pairs)
    # print("out",min_cost, best_pairs, len(avai_parts))
    return min_cost, best_pairs


def get_best_order(cost_n2n, mask_n2n, best_pairs, curr_pose):
    # print("in", best_pairs)
    punish_cost = 1000000
    curr_x = curr_pose[0]
    curr_y = curr_pose[1]
    best_order = []
    if len(best_pairs) <= 0:
        # print("None")
        c = cost_n2n[curr_x, -1, curr_y, -1]
        m = mask_n2n[curr_x, -1, curr_y, -1]
        if m == 0:
            c = punish_cost
        # print("out end", c, (-1,-1), curr_pose)
        return c, best_order
    else:
        # perm = permutations(best_pairs)
        min_cost = punish_cost
        for pairs in best_pairs:
            # print("pairs, \t", pairs[0], pairs[1])
            perm = permutations(pairs, 2)
            for pair in perm:
                # print("pair,",pair)
                # pair = p[0]
                target_x = pair[0]
                target_y = pair[1]
                c = cost_n2n[curr_x, target_x, curr_y, target_y]
                m = mask_n2n[curr_x, target_x, curr_y, target_y]
                # print("c", c)
                if m == 0:
                    continue

                pairs_left = best_pairs.copy()
                pairs_left.remove(pairs)
                future_c, order = get_best_order(cost_n2n, mask_n2n, pairs_left, pair)
                # print("back", future_c, order)
                cost = c + future_c

                if cost < min_cost:
                    min_cost = cost
                    best_order = order.copy()
                    best_order.append(pair)
                    # print("!!!!!!!!!!!!!1update", cost,best_order, curr_pose, c)
    # print("out",min_cost, best_order, len(best_pairs))

    return min_cost, best_order


def baseline_offline_brute_force(cost, mask, parts_mask):
    time0 = time.time()
    cost_node = cost['n']
    cost_node2node = cost['n2n']

    mask_node = mask['n']
    mask_node2node = mask['n2n']

    available_parts = list(np.arange(len(parts_mask)))
    remove_list = []
    for k in range(len(parts_mask)):
        if parts_mask[k] == 0:
            remove_list.append(available_parts[k])
    for rm in remove_list:
        available_parts.remove(rm)
    # print(parts_mask, available_parts)

    max_cost = 1000000
    min_cost = max_cost
    best_order = available_parts.copy()
    perm = permutations(available_parts)
    for p in perm:
        cost = get_cost_of_order(cost_node, cost_node2node, mask_node, mask_node2node, p)
        if cost < min_cost:
            min_cost = cost
            best_order = p

    time2 = time.time()
    print("total cost:\t", min_cost)
    print("best order:\n", best_order)
    print("total time used:\t",time2 - time0)

    return min_cost, best_order, time2 - time0

def calcul_cost(cost, mask, parts_mask, order, show = False):
    time0 = time.time()
    cost_node = cost['n']
    cost_node2node = cost['n2n']

    mask_node = mask['n']
    mask_node2node = mask['n2n']

    available_parts = list(np.arange(len(parts_mask)))
    remove_list = []
    for k in range(len(parts_mask)):
        if parts_mask[k] == 0:
            remove_list.append(available_parts[k])
    for rm in remove_list:
        available_parts.remove(rm)
    # print(parts_mask, available_parts)
    if len(order) < len(available_parts):
        return 1000000

    cost = get_cost_of_order(cost_node, cost_node2node, mask_node, mask_node2node, order, show = show)
    return cost

def get_cost_of_order(cost_node, cost_node2node, mask_node, mask_node2node, order, curr_pose=None, go_back=True, show = False):
    cost = 0
    max_cost = 1000000
    if len(order) < 1:
        return cost
    i = 0
    if curr_pose is None:
        x0 = -1
        y0 = -1
    else:
        x0 = curr_pose[0]
        y0 = curr_pose[1]
    if show:
        print("\n$$$$$$$$$$$$$$$$$$$$$")
        print("start:\t", x0, y0)
    while i < len(order):
        x_count = i
        y_count = i + 1
        if y_count >= len(order):
            break
        x = order[x_count]
        y = order[y_count]
        c1 = cost_node[x, y]
        c2 = cost_node2node[x0, x, y0, y]
        m1 = mask_node[x, y]
        m2 = mask_node2node[x0, x, y0, y]
        if show:
            print("new step:\t", x, y)
            print("availability:\t", m1, m2)
        if not m1 or not m2:
            if show:
                print("not available, fail solution")
            return max_cost
        if show:
            print("cost:\t", c1, c2)
        cost += c1 + c2

        x0 = x
        y0 = y
        i += 2
    if go_back:
        if show:
            print("go back, cost:\t", cost_node2node[x0, -1, y0, -1])
        cost += cost_node2node[x0, -1, y0, -1]
    if show:
        print("total cost:\t", cost)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    return cost


def baseline_online_MCTS(cost, mask, parts_mask, step=6):
    print("sssssss")
    look_forward_step = step
    time0 = time.time()
    cost_node = cost['n']
    cost_node2node = cost['n2n']

    mask_node = mask['n']
    mask_node2node = mask['n2n']

    available_parts = list(np.arange(len(parts_mask)))
    remove_list = []
    for k in range(len(parts_mask)):
        if parts_mask[k] == 0:
            remove_list.append(available_parts[k])
    for rm in remove_list:
        available_parts.remove(rm)

    # print(parts_mask, available_parts)
    best_order = []
    curr_pose = (-1,-1)
    while len(available_parts) > 1:
        c, c0, pair = MTCS(cost_node, cost_node2node, mask_node, mask_node2node, available_parts, curr_pose,
                       look_forward_step=look_forward_step)
        if pair is not None:
            print("take action:\t",pair[0], pair[1])
            x = pair[0]
            y = pair[1]
            best_order.append(x)
            best_order.append(y)
            available_parts.remove(x)
            available_parts.remove(y)
            print("step cost:\t", c0,"\tforward cost:\t", c)
            curr_pose = (pair)

        else:
            break


    min_cost = get_cost_of_order(cost_node, cost_node2node, mask_node, mask_node2node, best_order)
    best_order = tuple(best_order)
    time2 = time.time()
    if len(best_order) < len(available_parts):
        min_cost += 1000000
    print(" ")
    print("total cost:\t", min_cost)
    print("best order:\n", best_order)
    print("total time used:\t", time2 - time0)
    print("sssssss")
    return min_cost, best_order, time2 - time0


def MTCS(cost_node, cost_node2node, mask_node, mask_node2node, available_parts, curr_pose, look_forward_step=6):
    max_cost = 1000
    if len(available_parts) <= look_forward_step:
        look_forward_step = len(available_parts)
        go_back = True
    else:
        go_back = False

    best_pair = None
    min_cost = max_cost
    curr_cost = max_cost

    perm = permutations(available_parts, look_forward_step)
    for p in perm:
        cost = get_cost_of_order(cost_node, cost_node2node, mask_node, mask_node2node, p, curr_pose=curr_pose,
                                 go_back=go_back)

        if cost < min_cost:
            min_cost = cost
            best_pair = p
            curr_cost = cost_node[p[0],p[1]] + cost_node2node[curr_pose[0], p[0], curr_pose[1], p[1]]

    return min_cost, curr_cost, best_pair,
