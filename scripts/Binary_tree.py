import numpy as np

class TreeNode:
    """
    Class representing a node in a binary tree.
    
    Attributes
    ----------
    left : TreeNode or None
        Left child.
    right : TreeNode or None
        Right child.
    index : int or None
        Node index (can be set via inorder numbering).
    parent : TreeNode or None
        Parent node.
    is_on_leftmost_path : bool
        True if node lies on the leftmost path from root.
    is_on_rightmost_path : bool
        True if node lies on the rightmost path from root.
    depth : int
        Depth of the node in the tree.
    """
    def __init__(self):
        self.left = None
        self.right = None
        self.index = None
        self.parent = None
        self.is_on_leftmost_path = False
        self.is_on_rightmost_path = False
        self.depth = 0


def get_node_with_index(root: TreeNode, x: int) -> TreeNode:
    """
    Searches for a node with a given index in a binary search tree.
    
    Parameters
    ----------
    root : TreeNode
        Root of the BST.
    x : int
        Value to search for.
    
    Returns
    -------
    TreeNode or None
        Node with value x if found, else None.
    """
    if root is None:
        return None
    if root.index == x:
        return root
    elif x < root.index:
        return get_node_with_index(root.left, x)
    else:  # x > root.index
        return get_node_with_index(root.right, x)


def build_complete_binary_tree(n: int) -> TreeNode:
    """
    Builds a complete binary tree with n nodes (without indeces).
    
    Parameters
    ----------
    n : int
        Number of nodes.
    
    Returns
    -------
    TreeNode
        Root node of the tree.
    """
    if n == 0:
        return None
    
    nodes = [TreeNode() for _ in range(n)]
    for i in range(n):
        left_index = 2 * i + 1
        right_index = 2 * i + 2
        nodes[i].depth = int(np.floor(np.log2(i + 1)))
        if left_index < n:
            nodes[i].left = nodes[left_index]
            nodes[left_index].parent = nodes[i]
        if right_index < n:
            nodes[i].right = nodes[right_index]
            nodes[right_index].parent = nodes[i]
    
    return nodes[0]  # return root


def mark_leftmost_rightmost_paths(root: TreeNode):
    """
    Marks nodes that lie on the leftmost and rightmost paths from root.
    
    Parameters
    ----------
    root : TreeNode
        Root node of the tree.
    """
    # Leftmost path
    node = root
    while node:
        node.is_on_leftmost_path = True
        node = node.left
    
    # Rightmost path
    node = root
    while node:
        node.is_on_rightmost_path = True
        node = node.right


def inorder_numbering(node: TreeNode, counter: int) -> int:
    """
    Assigns inorder numbers to the nodes of the tree recursively.
    
    Parameters
    ----------
    node : TreeNode
        Current node.
    counter : int
        Current numbering counter.
    
    Returns
    -------
    int
        Updated counter after numbering this subtree.
    """
    if node is None:
        return counter
    counter = inorder_numbering(node.left, counter)
    node.index = counter
    counter += 1
    counter = inorder_numbering(node.right, counter)
    return counter


def construct_inorder_numbered_tree(n: int) -> TreeNode:
    """
    Constructs a complete binary tree with n nodes, numbers them
    in inorder, and marks the leftmost and rightmost paths.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    
    Returns
    -------
    TreeNode
        Root node of the constructed tree.
    """
    root = build_complete_binary_tree(n)
    inorder_numbering(root, 0)
    mark_leftmost_rightmost_paths(root)
    return root
