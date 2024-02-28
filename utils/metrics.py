import torchvision, torch
import numpy as np
import scipy
from .constants import ESSENTIALCLSSES

def compute_matching_score(nodes_list, probs, edges_list, gt_list, curve=False, clsname2id=ESSENTIALCLSSES, edge_threshold=0.5):
    """
    Calculating matching F1 score / Precision-Recall curve.

    node_list: a list of mung.node.Node, representing predicted nodes
    probs: a numpy array of shape (len(node_list), number_of_classes), representing probability distributions for each node.
    edges_list: a list of triplet (node1 idx, node2 idx, edge score), no repetition.
    gt_list: a list of mung.node.Node, representing ground truth nodes
    curve: whether to return a precesion-recall curve
    classname2id: the dict from class name to class id. Default to ESSENTIALCLSSES
    edge_threshold: the threshold for edge scores, used for calculating F1 score with curve=False.

    When curve = False, return the F1 score
    When curve = True, return two lists (precision_list, recall_list)
    """
    # Get box matrix from nodes_list
    boxes = [] 
    for node in nodes_list:
        boxes.append((node.left, node.top, node.right, node.bottom))

    # Filter invalid classes in gt_list
    gt_list_ = [node for node in gt_list if node.class_name in clsname2id]

    # Get box matrix and prob matrix from gt_list
    boxes_g = []
    prob_matrix = []
    id2idx = {}
    for idx, node in enumerate(gt_list_):
        id2idx[node.id] = idx
        prob_matrix.append(probs[:, clsname2id[node.class_name]])
        boxes_g.append((node.left, node.top, node.right, node.bottom))
    prob_matrix = np.stack(prob_matrix).transpose()

    # Computer pairwise iou
    box_matrix = torchvision.ops.box_iou(torch.tensor(boxes), torch.tensor(boxes_g)).numpy()

    # Construct and solve matching problem
    cost_matrix = np.multiply(box_matrix, prob_matrix)
    row_indices, col_indices = scipy.optimize.linear_sum_assignment(-cost_matrix)
    match_a = {row_idx:col_idx for row_idx, col_idx in zip(row_indices, col_indices) if cost_matrix[row_idx, col_idx] > 0.05}

    all_edges = set()
    for i, node in enumerate(gt_list_):
        for to_id in node.outlinks:
            if to_id not in id2idx:
                continue
            j = id2idx[to_id] 
            all_edges.add((i, j) if i < j else (j, i))
    TP, FP, TN, FN = 0, 0, len(gt_list_)*(len(gt_list_)-1)//2-len(all_edges), len(all_edges)
    precision, recall = [], []
    for i, j, _ in sorted(edges_list, key=lambda x:x[2], reverse=True):
        if not curve and _ < edge_threshold:
            break
        if i not in match_a or j not in match_a: # Redudant nodes
            continue
        _i, _j = match_a[i], match_a[j]
        if ((_i, _j) if _i < _j else (_j, _i)) in all_edges:
            TP += 1
            FN -= 1
        else:
            FP += 1
            TN -= 1
        precision.append(TP/(TP+FP))
        recall.append(TP/(TP+FN))
    if curve:
        return precision, recall
    assert len(precision) > 0, "All edge score < edge_threshold"
    return 2 * (precision[-1]*recall[-1])/(precision[-1]+recall[-1])
    

if __name__ == "__main__":
    from mung.io import read_nodes_from_file
    import numpy as np
    nodes_list = read_nodes_from_file("/local1/MUSCIMA/v2.0_gen/data/annotations/CVC-MUSCIMA_W-01_N-10_D-ideal.xml")
    probs = np.load("/local1/MUSCIMA/v2.0_gen/data/annotations/CVC-MUSCIMA_W-01_N-10_D-ideal.npy")
    gt_list = read_nodes_from_file("/local1/MUSCIMA/v2.0/data/annotations/CVC-MUSCIMA_W-01_N-10_D-ideal.xml")
    assert len(nodes_list[0].data['unoutlinks']) == 73 # The false edges
    id2idx = {node.id: i for i, node in enumerate(nodes_list)} ## In this example, id and idx are the same
    edges_list = []
    for i, node in enumerate(nodes_list):
        for j in node.outlinks:
            if j in id2idx:
                edges_list.append((i, j, 1.0))
    precision, recall = compute_matching_score(nodes_list, probs, edges_list, gt_list, curve=True) # Precision all 1.0, Recall from 0.0 to 0.8797
    assert np.all(np.equal(precision, 1.0))
    print(compute_matching_score(nodes_list, probs, edges_list, gt_list)) # Reference value : 0.9359756097560976