def matching_score(nodes_list, edges_list, gt_list, curve=False):
    """
    Calculating matching F1 score / Precision-Recall curve.

    node_list: a list of mung.node.Node, representing predicted nodes
    edges_list: a list of triplet (node1 idx, node2 idx, edge score)
    gt_list: a list of mung.node.Node, representing ground truth nodes
    curve: whether to return a precesion-recall curve

    When curve = False, return the F1 score
    When curve = True, return two lists (precision_list, recall_list)
    """
    pass