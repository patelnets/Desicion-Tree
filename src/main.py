import numpy as np
from treeplot import draw_tree
import sys
import getopt

######DO NOT EDIT HERE. EDIT BELOW. SEARCH DEFAULT INPUT#######
clean_data = np.loadtxt("../rec/clean_dataset.txt")
noisy_data = np.loadtxt("../rec/noisy_dataset.txt")

number_of_rooms = 4
number_of_wifi = 7
depth = 0
nodeid = 0

pruned_nodes = 0

# Calculates the average entropy of the dataset
def average_entropy(dataset):
    if len(dataset) == 0:
        raise RuntimeError("Got emtpy dataset as input")

    number_of_samples_in_rooms = np.zeros(number_of_rooms)

    for row in dataset:
        number_of_samples_in_rooms[int(row[7]) - 1] += 1
    probability_of_room = number_of_samples_in_rooms / len(dataset)
    average_entropy = 0
    for probability in probability_of_room:
        if probability != 0:
            average_entropy = average_entropy - \
                (probability * np.log2(probability))

    return average_entropy

# TODO:
def get_gain(s_all, s_left, s_right):
    return average_entropy(s_all) - remainder(s_left, s_right)

# TODO:
def remainder(s_left, s_right):
    total = len(s_left) + len(s_right)
    return ((len(s_left) / total) * average_entropy(s_left)) + (
        (len(s_right) / total) * average_entropy(s_right))

# Sorts the dataset by a specific attribute
def sort_by_attribute(dataset, attribute):
    return dataset[dataset[:, attribute].argsort()]

# Finds the best split point of a dataset
def find_split(dataset):
    gain = -1
    att_value = None
    att = None
    s_left = None
    s_right = None

    # For each attribute we sort by its value and try to find the split
    # point that nets the best result
    for attribute in range(number_of_wifi):
        dataset = sort_by_attribute(dataset, attribute)
        for row in range(len(dataset) - 1):
            if dataset[row][attribute] != dataset[row + 1][attribute]:
                gain_tmp = get_gain(
                    dataset, dataset[:row + 1], dataset[row + 1:])
                if gain_tmp > gain:
                    gain = gain_tmp
                    att_value = (dataset[row][attribute]
                                 + dataset[row + 1][attribute]) / 2
                    att = attribute
                    s_left = dataset[:row + 1]
                    s_right = dataset[row + 1:]

    dataset = sort_by_attribute(dataset, att)
    # Returns the which attribute we split by, its value and the sides of the
    # split dataset
    return att, att_value, s_left, s_right


# Checks if all the elements in the dataset have the same label
def all_samples_same_label(dataset):
    label = dataset[0][-1]
    for sample in dataset:
        if sample[-1] != label:
            return False

    return True

# Trains the decision tree based on the given training_dataset
def decision_tree_learning(training_dataset):
    # If all the sample in the subset have the same label, than we can insert
    # a leaf with that label here
    if all_samples_same_label(training_dataset):
        return {'value': training_dataset[0][-1]}, 0, 1
    # Otherwise we find the best split point, to divide the data by
    else:
        attribute, value, lset, rset = find_split(training_dataset)
        node = {'attribute': attribute, 'value': value}

        node['left'], l_depth, lleafs = decision_tree_learning(lset)
        node['right'], r_depth, rleafs = decision_tree_learning(rset)

        return node, max(l_depth, r_depth) + 1, rleafs + lleafs

# Calculates the depth, total number of nodes and number of leaves
def calculate_depth(node):
    if 'attribute' not in node:
        return 0, 1, 1
    else:
        l_depth, lnodes, lleafs = calculate_depth(node['left'])
        r_depth, rnodes, rleafs = calculate_depth(node['right'])
        return max(l_depth, r_depth) + 1, lnodes + rnodes + 1, rleafs + lleafs

# Runs a value through the decision tree to see what label gets outputted
def evaluate_value_from_tree(value, node):
    if 'attribute' in node:
        if value[node['attribute']] < node['value']:
            return evaluate_value_from_tree(value, node['left'])
        else:
            return evaluate_value_from_tree(value, node['right'])
    else:
        return node['value']

# Calculates the desired metrics for the trained_tree, based on the given test_set
def evaluate(test_set, trained_tree):
    results = []
    precision = []
    recall = []
    f1_measure = []

    confusion_matrix = np.zeros(shape=(4, 4), dtype=int)
    for value in test_set:
        result = evaluate_value_from_tree(value, trained_tree)
        confusion_matrix[int(value[-1] - 1)][int(result - 1)] += 1

    for room in range(0, number_of_rooms):
        true_positive = confusion_matrix[room][room]
        false_positive = sum(confusion_matrix[:, room]) - true_positive
        false_negative = sum(confusion_matrix[room]) - true_positive
        true_negative = np.trace(confusion_matrix) - true_positive

        if true_positive == 0:
            precision.append(0)
            recall.append(0)
            f1_measure.append(0)
        else:
            precision.append(true_positive / (true_positive + false_positive))
            recall.append(true_positive / (true_positive + false_negative))
            f1_measure.append(
                2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1]))
    class_rate = np.trace((confusion_matrix) / len(test_set))

    return confusion_matrix, precision, recall, f1_measure, class_rate

# Calculates the average confusion matrix based on a list of confusion matrices
def average_matrix(list_of_matrices):
    avg_confusion_matrix = np.zeros(list_of_matrices[0].shape, dtype=float)

    for row in range(np.size(list_of_matrices[0], 0)):
        for col in range(np.size(list_of_matrices[0], 1)):
            sum = 0
            for matrix in list_of_matrices:
                sum = sum + matrix[row][col]
            avg_confusion_matrix[row][col] = sum / float(len(list_of_matrices))
    return avg_confusion_matrix

# Calculates the per class average
def calculate_avg_per_class(metric_all):
    metric_avg = []
    for i in range(len(metric_all[0])):
        sum = 0
        for j in range(len(metric_all)):
            sum = sum + metric_all[j][i]
        metric_avg.append(sum / len(metric_all))
    return metric_avg

# Performs k-fold validation
def k_fold_validation(k, dataset, pruning=False):
    # np.random.seed(0)
    np.random.shuffle(dataset)
    dataset_length = len(dataset)
    dataset = np.reshape(dataset, (k, int(dataset_length / k), 8))
    confusion_matrix_all = []
    precision_all = []
    recall_all = []
    f1_measure_all = []
    class_rate_all = []
    max_depth_all = []
    for i in range(k):
        dataset_copy = dataset
        test_set = dataset_copy[i]
        training_and_validation_set = np.delete(dataset_copy, i, 0)
        for j in range(k - 1):
            training_and_validation_set_copy = training_and_validation_set
            if pruning:
                validation_set = training_and_validation_set_copy[j]
                training_set = np.delete(
                    training_and_validation_set_copy, j, 0)
                training_set = np.reshape(
                    training_set, (dataset_length - int(dataset_length / k) * 2, 8))
            else:
                training_set = np.reshape(
                    training_and_validation_set_copy, (dataset_length - int(dataset_length / k), 8))
            treeRoot, depth, _ = decision_tree_learning(training_set)
            depth, _, _ = calculate_depth(treeRoot)
            if pruning:
                prune_until_no_improvement(validation_set, treeRoot)
                depth, _, _ = calculate_depth(treeRoot)

            confusion_matrix, precision, recall, f1_measure, class_rate = evaluate(
                test_set, treeRoot)
            confusion_matrix_all.append(confusion_matrix)
            precision_all.append(precision)
            recall_all.append(recall)
            f1_measure_all.append(f1_measure)
            class_rate_all.append(class_rate)
            max_depth_all.append(depth)

            print("Cross validation results found round {round} of {total_rounds}:\n".format(
                round=(i * (k - 1)) + j + 1, total_rounds=k * (k - 1)))

    return average_matrix(confusion_matrix_all), calculate_avg_per_class(
        precision_all), calculate_avg_per_class(recall_all), calculate_avg_per_class(
        f1_measure_all), np.mean(class_rate_all), np.mean(max_depth_all)

# Helper function to print cross validation results
def print_cross_validation_results(confusion_matrix_all, precision_all, recall_all, f1_measure_all,
                                   class_rate_all):
    for i in range(len(confusion_matrix_all[0])):
        precision_sum = 0
        recall_sum = 0
        f1_measure_sum = 0
        class_rate_sum = 0
        for j in range(len(confusion_matrix_all)):
            precision_sum += precision_all[j][i]
            recall_sum += recall_all[j][i]
            f1_measure_sum += f1_measure_all[j][i]
            class_rate_sum += class_rate_all[j]
        print("The average precision for for room {room} is {precision_avg}".format(
            room=i + 1, precision_avg=precision_sum / len(confusion_matrix_all)))
        print("The average recall for for room {room} is {recall_avg}".format(
            room=i + 1, recall_avg=recall_sum / len(confusion_matrix_all)))
        print("The average f1 measure for for room {room} is {f1_measure_avg}".format(
            room=i + 1, f1_measure_avg=f1_measure_sum / len(confusion_matrix_all)))
        print("The average class rate for for room {room} is {class_rate_avg}".format(
            room=i + 1, class_rate_avg=class_rate_sum / len(confusion_matrix_all)))

# Checks if the curret node is leaf
def is_leaf(node):
    return 'attribute' not in node

# Checks if the current node is the parent of two leaves
def has_two_leaves(node):
    return is_leaf(node['left']) and is_leaf(node['right'])


# Checks to see if the current node needs to be pruned from the tree and if so
# replaces it with the child leaf that nets the best improvement
def prune_node(validation_set, tree, node):
    accuracy = evaluate(validation_set, tree)[-1]
    original_attribute = node.pop('attribute', None)
    left_child = node.pop('left', None)
    right_child = node.pop('right', None)
    if (original_attribute is None or left_child is None or right_child is None):
        raise RuntimeError("None got ")
    original_value = node['value']

    node['value'] = left_child['value']
    left_prune_accuracy = evaluate(validation_set, tree)[-1]

    node['value'] = right_child['value']
    right_prune_accuracy = evaluate(validation_set, tree)[-1]

    if max(left_prune_accuracy, right_prune_accuracy) >= accuracy:
        if left_prune_accuracy > right_prune_accuracy:
            node['value'] = left_child['value']
            return True
        else:
            node['value'] = right_child['value']
            return True
    else:
        node['attribute'] = original_attribute
        node['value'] = original_value
        node['left'] = left_child
        node['right'] = right_child
        return False

# Runs the pruning algorithm over the desired tree
def prune_tree(validation_set, tree_root, node):
    if not is_leaf(node):
        if has_two_leaves(node):
            new_value_for_pruned_node = 0
            if prune_node(validation_set, tree_root, node):
                return 2
        else:
            return prune_tree(validation_set, tree_root, node['left']) \
                + prune_tree(validation_set, tree_root, node['right'])
    return 0

# Repeatedly runs the pruning algorithm until there are no node that need to be
# pruned anymore
def prune_until_no_improvement(validation_data, tree):
    pruned_nodes = total_pruned = prune_tree(validation_data, tree, tree)
    counter = 1
    while pruned_nodes > 0:
        counter = counter + 1
        pruned_nodes = prune_tree(validation_data, tree, tree)
        total_pruned += pruned_nodes
    print("Total nodes pruned ", total_pruned)
    # print("Number of times Pruned nodes called: ", counter)

# Helper function to run and print the data from kfold validation
def run_kfold(dataset, pruning, set=0):
    print('=' * 30)
    confusion_matrix_avg, precision_avg, recall_avg, f1_measure_avg, class_rate_avg, max_depth_avg = k_fold_validation(
        10, dataset, pruning=pruning)

    if set == 0:
        print("Metrics for dataset, where pruning is ", pruning)
    elif set == 1:
        print("Metrics for clean dataset, where pruning is ", pruning)
    else:
        print("Metrics for noisy dataset, where pruning is ", pruning)

    print("Confusion matrix Average: {confusion_matrix_avg}".format(
        confusion_matrix_avg=confusion_matrix_avg))
    for i in range(len(precision_avg)):
        print("For class {i}: ".format(i=i + 1))
        print("The Precision Average : {precision_avg}".format(
            precision_avg=precision_avg[i]))
        print("The Recall Average: {recall_avg}".format(
            recall_avg=recall_avg[i]))
        print("The F1 Measure Average: {f1_measure_avg}".format(
            f1_measure_avg=f1_measure_avg[i]))
        print("Class Rate  Average: {class_rate_avg}".format(
            class_rate_avg=class_rate_avg))
        print("Max Depth Average: {max_depth_avg}".format(
            max_depth_avg=max_depth_avg))


def main(argv):
    ######DEFAULT INPUT FILE EDIT HERE TO LOAD CUSTOM DEFAULT DATA#######
    inputfile = "../rec/clean_dataset.txt"
    single = True
    draw = True
    prune = False
    interactive = False
    all = False

    try:
        opts, args = getopt.getopt(
            argv, "hcni:r:p",
            ["input=", "runmode=", "help", "clean", "noisy", "drawoff", "prune"])
    except getopt.GetoptError:
        print("Invalid option!")
        print('python test.py -h for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("python3 main.py <-h or --help> for help")
            print("python3 main.py <-i or --input=> <filepath> to set input file")
            print(
                "python3 main.py <-c or --clean> to run on the clean dataset, this is the default running mode")
            print("python3 main.py <-n or --noisy> to run on the noisy dataset")
            print(
                "python3 main.py <-r or --runmode=> <single or kfold or kfoldall> to run in single or kfold mode. Default mode is single")
            print(
                "Running kfoldall runs all four possible combination of clean or noisy and pruning or no pruning")
            print(
                "python3 main.py <-p or --prune> to enable pruning. Default is off, single mode option only. Saves to diagrams")
            print(
                "A diagram is produced only when running in single mode at the following location ../out/diagram.pdf")
            print("python3 main.py <--drawoff> to disable diagram rendering")
            print(
                "\nSample commanand:\npython3 main.py --n --runmode=single --prune")
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-r", "--runmode"):
            if arg in ("single", "kfold", "kfoldall"):
                single = arg == "single"
                all = arg == "kfoldall"
            else:
                print('Unrecognized runmode. Run modes single, kfold or kfoldall')
                sys.exit(2)
        elif opt in ("-c", "--clean"):
            inputfile = "../rec/clean_dataset.txt"
        elif opt in ("-n", "--noisy"):
            inputfile = "../rec/noisy_dataset.txt"
        elif opt == "--drawoff":
            draw = False
        elif opt in ("-p", "--prune"):
            prune = True

    print("Input file is ", inputfile)
    dataset = np.loadtxt(inputfile)
    if single is True:
        np.random.shuffle(dataset)
        test_set = dataset[-200:]
        learn_set = dataset[200:-200]
        valid_set = dataset[0:200]
        root, _, _ = decision_tree_learning(learn_set)
        depth, nodes, leaves = calculate_depth(root)
        print("The depth is ", depth, " and the tree has ",
              nodes, " nodes and ", leaves, " leaves\n")
        print("Initial accuracy on test set: ", evaluate(test_set, root)[-1])
        print("Initial accuracy on valid set :",
              evaluate(valid_set, root)[-1], "\n")
        if draw is True:
            draw_tree(root, depth, leaves, interactive=interactive)
        if prune is True:
            prune_until_no_improvement(valid_set, root)
            depth, nodes, leaves = calculate_depth(root)
            print("The depth is ", depth, " and the tree has ",
                  nodes, " nodes and ", leaves, " leaves\n")
            print("After pruning accuracy on test set: ",
                  evaluate(test_set, root)[-1])
            print("After pruning accuracy on valid set :",
                  evaluate(valid_set, root)[-1], "\n")
            if draw is True:
                draw_tree(root, depth, leaves, after=True)
    else:
        if all is True:
            run_kfold(clean_data, False, 1)
            run_kfold(clean_data, True, 1)
            run_kfold(noisy_data, False, 2)
            run_kfold(noisy_data, True, 2)
        else:
            if len(dataset) % 10 != 0:
                dataset = dataset[0:int(len(dataset / 10)) * 10]
            run_kfold(dataset, prune)


if __name__ == "__main__":
    main(sys.argv[1:])
