import random

from error_align.utils import OpType, OP_TYPE_COMBO_MAP


class Node:
    """
    Node in the optimal alignment graph corresponding to the index (i, j) in the backtrace matrix.
    """

    def __init__(self, hyp_idx, ref_idx) -> None:
        """
        Initialize the node at index (i, j).
        """

        self.hyp_idx = hyp_idx
        self.ref_idx = ref_idx
        self.children = {}
        self.parents = {}
        
        # Used to count the number of paths going through this node.
        self._bwd_node_count = 0
        self._fwd_node_count = 0
        
        # Used to store the operations that lead to this node.
        self._ingoing_edge_counts = {}
        self._outgoing_edge_counts = {}

    @property
    def index(self):
        return (self.hyp_idx, self.ref_idx)
    
    @property
    def offset_index(self):
        """
        Get the offset index of the node so indices mathces the hypothesis and reference strings.
        
        Returns:
            tuple[int, int]: The offset index of the node in the backtrace matrix.
        """
        return (self.hyp_idx - 1, self.ref_idx - 1)

    @property
    def number_of_paths(self):
        return self._bwd_node_count * self._fwd_node_count
    
    def number_of_ingoing_paths_via(self, op_type: OpType):
        """
        Get the number of paths going through this node via the given operation type.
        
        Args:
            op_type (OpType): The operation type to check.
        """
        if op_type not in self.parents:
            return 0
        return self._ingoing_edge_counts[op_type] * self.parents[op_type]._outgoing_edge_counts[op_type]
    
    def number_of_outgoing_paths_via(self, op_type: OpType):
        """
        Get the number of paths going through this node via the given operation type.
        
        Args:
            op_type (OpType): The operation type to check.
        """
        if op_type not in self.children:
            return 0
        return self._outgoing_edge_counts[op_type] * self.children[op_type]._ingoing_edge_counts[op_type]
    
    @property
    def is_terminal(self):
        """
        Check if the node is a terminal node (i.e., it has no children).
        """
        return len(self.children) == 0
    
    @property
    def is_root(self):
        """
        Check if the node is a root node (i.e., it has no parents).
        """
        return len(self.parents) == 0


class OptimalAlignmentGraph:
    """
    Optimal alignment graph.
    """

    def __init__(self, B) -> None:
        """
        Create a graph from the backtrace matrix.
        """

        self.hyp_dim = B.shape[0]
        self.ref_dim = B.shape[1]
        self.hyp_max_idx = self.hyp_dim - 1
        self.ref_max_idx = self.ref_dim - 1
        self.B = B

        self._nodes = None
        # terminal_node = Node(self.hyp_max_idx, self.ref_max_idx)
        # self.nodes = {terminal_node.index: terminal_node}

        # # Traverse nodes in reverse topological order to add parents.
        # for index in self._iter_topological_order(reverse=True):
        #     if index in self.nodes:
        #         self._add_parents_from_backtrace(index)
        
        # # Sort nodes by their indices to ensure topological order.
        # self.nodes = dict(sorted(self.nodes.items(), key=lambda item: (item[0][0], item[0][1])))
        # self._set_path_and_node_counts() # TODO: Consider to do this lazily. Shouldn't be too expensive.

    @property
    def nodes(self) -> dict[tuple[int, int], Node]:
        """
        Get the nodes in the graph.
        
        Returns:
            dict: A dictionary of nodes indexed by their (hyp_idx, ref_idx).
        """
        if self._nodes is None:
            terminal_node = Node(self.hyp_max_idx, self.ref_max_idx)
            self._nodes = {terminal_node.index: terminal_node}

            # Traverse nodes in reverse topological order to add parents.
            for index in self._iter_topological_order(reverse=True):
                if index in self._nodes and index != (0, 0):
                    self._add_parents_from_backtrace(index)
            
            # Sort nodes by their indices to ensure topological order.
            self._nodes = dict(sorted(self._nodes.items(), key=lambda item: (item[0][0], item[0][1])))
            self._set_path_and_node_counts() # TODO: Consider to do this lazily. Shouldn't be too expensive.
        
        return self._nodes
        
    @property
    def number_of_paths(self):
        """
        Count the number of paths in the graph.

        Returns:
            int: The number of paths.
        """
        return self.get_node(0, 0)._bwd_node_count

    def get_node(self, hyp_idx, ref_idx):
        """
        Get the node at the given index.

        Args:
            hyp_idx (int): Hyp/row index.
            ref_idx (int): Ref/column index.
        """
        return self.nodes[(hyp_idx, ref_idx)]
    
    def get_node_indices(self):
        """
        Get the outgoing transitions from each node in the graph.

        Returns:
            dict: A dictionary where keys are node indices and values are lists of outgoing transitions.
        """
        transitions = set()
        for node in self.nodes.values():
            transitions.add(node.index)
        return transitions
    
    def get_path(self, sample=False):
        """
        Get a path through the graph.

        Args:
            sample (bool): If True, sample a path randomly based on the transition probabilities. Otherwise, return 
            the first path deterministically.
        
        Returns:
            list[Node]: A list of nodes representing the path.
        """
        node = self.get_node(0, 0)
        assert node.is_root, "The node at (-1, -1) was expected to be a root node."
        
        path = []
        while not node.is_terminal:
            if sample:
                op_type = random.choice(list(node.children.keys()))
            else:
                op_type = next(iter(node.children.keys()))
            node = node.children[op_type]
            path.append((op_type, node))
        
        return path

    def _parent_index_from_op_type(self, hyp_idx, ref_idx, op_type):
        """
        Create a parent node based on the index of the current node and the operation type.

        Args:
            hyp_idx (int): Parent row index.
            ref_idx (int): Parent column index.
            op_type (OpType): The operation type.
        """
        hyp_idx = hyp_idx - 1 if op_type != OpType.INSERT else hyp_idx
        ref_idx = ref_idx - 1 if op_type != OpType.DELETE else ref_idx
        if (hyp_idx, ref_idx) not in self._nodes:
            self._nodes[(hyp_idx, ref_idx)] = Node(hyp_idx, ref_idx)
        return self._nodes[(hyp_idx, ref_idx)]
    
    def _iter_topological_order(self, reverse=False):
        """
        Iterate through the nodes in topological order.

        Args:
            reversed (bool): If True, iterate in reverse order.
        """
        if reversed:
            for i in reversed(range(self.hyp_dim)):
                for j in reversed(range(self.ref_dim)):
                    yield (i, j)
        else:
            for i in range(self.hyp_dim):
                for j in range(self.ref_dim):
                    yield (i, j)

    def _add_parents_from_backtrace(self, index):
        """
        Add parents to the node at the given index based on the backtrace matrix.

        Args:
            index (tuple[int, int]): The index of the node in the backtrace matrix.
        """
        node = self._nodes.get(index, None)
        if node is None:
            raise ValueError(f"Node at index {index} does not exist in the graph.")
        
        op_type_combo_code = self.B[node.index]
        op_type_combo = OP_TYPE_COMBO_MAP[op_type_combo_code]

        for op_type in op_type_combo:
            parent_node = self._parent_index_from_op_type(*node.index, op_type)
            node.parents[op_type] = parent_node
            parent_node.children[op_type] = node

    def _set_path_and_node_counts(self):
        """
        Count the number of paths going through any node in the graph using the forward-backward algorithm.
        """
        
        ordered_nodes = list(self.nodes.values())
        root_node = ordered_nodes[0]
        terminal_node = ordered_nodes[-1]
        assert root_node.is_root, "The first node must be a root node."
        assert terminal_node.is_terminal, "The last node must be a terminal node."
        
        # Forward pass
        ordered_nodes[0]._fwd_node_count = 1
        for node in ordered_nodes[1:]:
            for op_type, parent in node.parents.items():
                node._fwd_node_count += parent._fwd_node_count
                node._ingoing_edge_counts[op_type] = parent._fwd_node_count
        
        # Backward pass
        ordered_nodes[-1]._bwd_node_count = 1
        for node in reversed(ordered_nodes[:-1]):
            for op_type, child in node.children.items():
                node._bwd_node_count += child._bwd_node_count
                node._outgoing_edge_counts[op_type] = child._bwd_node_count

        # Validate that the number of forward and backward paths are equal
        assert root_node._bwd_node_count == terminal_node._fwd_node_count, \
            "The number of forward and backward paths must be equal."    
    
    
    
    
    
    
    
    
    
    
    
    
    
