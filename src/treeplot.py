import matplotlib.pyplot as plt

# Calculates the number of nodes that are on each depth level
def depth_helper(node, current_depth):
    depth_count[current_depth] += 1

    if 'attribute' in node:
        depth_helper(node['left'], current_depth + 1)
        depth_helper(node['right'], current_depth + 1)

# Draws a diagram of the given tree
def draw_tree(root, depth, leafs, unit_size=200, height_size=100, after=False, interactive=False):
    arrowstyle = dict(arrowstyle="-", fc="w")
    node_bbox = dict(boxstyle="round", fc="w")
    leaf_colors = ["g", "b", "r", "y"]
    padding = 10

    global depth_count
    queue = [(root, 0)]

    if leafs < 20:
        leafs = 20

    unit_size = leafs / 1.5
    height_size = leafs

    depth_count = [0] * (depth + 1)
    depth_helper(root, 0)
    max_width = max(depth_count)
    center = max_width / 2 * unit_size

    # Calculates the x coordinate of the starting point for each level
    level_start = [padding + center - (x / 2) * unit_size for x in depth_count]

    fig = plt.figure(figsize=(leafs / 2, depth * 3))
    fig.clf()
    plot = fig.add_subplot(111)
    plot.set_aspect(1)

    level = 0
    index = 0
    leafs = 0

    while len(queue) > 0:
        (node, node_depth) = queue.pop(0)

        if level != node_depth:
            index = 0
            leafs = 0
            level += 1

        node_coords = (level_start[level] + index * unit_size,
                       (depth - level) * height_size)

        if 'attribute' in node:
            plot.annotate("", xy=(level_start[level + 1] + (2 * (index - leafs)) * unit_size, (depth - level - 1) * height_size), xytext=(node_coords),
                          va="center", ha="center", bbox=node_bbox, arrowprops=arrowstyle)
            queue.append((node['left'], level + 1))

            plot.annotate("", xy=(level_start[level + 1] + (2 * (index - leafs) + 1) * unit_size, (depth - level - 1) * height_size), xytext=(node_coords),
                          va="center", ha="center", bbox=node_bbox, arrowprops=arrowstyle)
            queue.append((node['right'], level + 1))

            plot.annotate("X" + str(node['attribute']) + " < " + str(
                node['value']), xy=node_coords, va="center", ha="center", bbox=node_bbox)
        else:
            leafs += 1
            plot.annotate(
                "leaf:" + str(node['value']), xy=node_coords, va="center", ha="center", bbox=dict(boxstyle="round", fc=leaf_colors[int(node['value']) - 1]))
        index += 1

    plot.set_xlim(0, 3 * padding + max_width * unit_size)
    plot.set_ylim(0, 3 * padding + depth * height_size)

    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plot.set_yticklabels([])
    plot.set_xticklabels([])

    if after is True:
        plt.savefig('../out/diagram_after_pruning.pdf', format='pdf')
    else:
        plt.savefig('../out/diagram.pdf', format='pdf')

    if interactive is True:
        plt.draw()
        plt.show()
