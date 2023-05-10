from typing import Union, Any
import numpy as np
import turtle


def _check_probability_vector(p):
    if abs(sum(p) - 1) >= 1e-10:
        raise ValueError("probability vector does not sum up to 1")
    if any(pi <= -1e-16 for pi in p):
        raise ValueError("probability vector contains negative entries")
    return True


def _check_stopping_time(n, t):
    if t > n:
        raise ValueError("stopping time greater than number of stages")
    return True


class ScenarioTree:
    """
    Scenario tree creation and visualisation
    """

    def __init__(self, stages, ancestors, probability, w_values=None, is_markovian=False):
        """
        :param stages: integer number of total tree stages (N+1)
        :param ancestors: array where `array position=node number` and `value at position=node ancestor`
        :param probability: array where `array position=node number` and `value at position=probability node occurs`
        :param w_values: array where `array position=node number` and `value at position=value of w`

        Note: avoid using this constructor directly; use a factory instead
        """
        self.__is_markovian = is_markovian
        self.__stages = stages
        self.__ancestors = ancestors
        self.__probability = probability
        self.__w_idx = w_values
        # this will be updated later (the user doesn't need to provide it)
        self.__children = None
        self.__data = None  # really, any data associated with the nodes of the tree
        self.__update_children()
        self.__allocate_data()

    def probability(self):
        return self.__probability

    def __update_children(self):
        self.__children = []
        for i in range(self.num_nonleaf_nodes):
            children_of_i = np.where(self.__ancestors == i)
            self.__children += children_of_i

    def __allocate_data(self):
        self.__data = np.empty(shape=(self.num_nodes, ), dtype=dict)

    def get_data_at_node(self, node_idx):
        """
        :param node_idx: index of node, or range of indices
        :return: stored data

        Returns `None` if no data are stored at the given node
        """
        return self.__data[node_idx]

    def set_data_at_node(self, node_idx, data_dict: dict):
        """
        :param node_idx: node index
        :param data_dict: a dictionary with the data to be stored at the above node
        :return: nothing
        """
        self.__data[node_idx] = data_dict

    @property
    def is_markovian(self):
        return self.__is_markovian

    @property
    def num_nonleaf_nodes(self):
        return np.sum(self.__stages < (self.num_stages - 1))

    @property
    def num_nodes(self):
        """
        :return: total number of nodes of the tree
        """
        return len(self.__ancestors)

    @property
    def num_stages(self):
        """
        :return: number of stages including zero stage
        """
        return self.__stages[-1] + 1

    def ancestor_of(self, node_idx):
        """
        :param node_idx: node index
        :return: index of ancestor node
        """
        return self.__ancestors[node_idx]

    def children_of(self, node_idx):
        """
        :param node_idx: node index
        :return: list of children of given node
        """
        return self.__children[node_idx]

    def stage_of(self, node_idx):
        """
        :param node_idx: node index
        :return: stage of given node
        """
        if node_idx < 0:
            raise ValueError("node_idx cannot be <0")
        return self.__stages[node_idx]

    def num_possible_events(self):
        """
        :return: max number of events at any node
        """
        return len(self.__w_idx)

    def event_at_node(self, node_idx):
        """
        :param node_idx: node index
        :return: the event (disturbance) `w` that caused `node_idx` to exist
        """
        return self.__w_idx[node_idx]

    def nodes_at_stage(self, stage_idx):
        """
        :param stage_idx: index of stage
        :return: array of node indices at given stage
        """
        return np.where(self.__stages == stage_idx)[0]

    def probability_of_node(self, node_idx):
        """
        :param node_idx: node index
        :return: probability to visit the given node
        """
        return self.__probability[node_idx]

    def siblings_of_node(self, node_idx):
        """
        :param node_idx: node index
        :return: array of siblings of given node (including the given node)
        """
        if node_idx == 0:
            return [0]
        return self.children_of(self.ancestor_of(node_idx))

    def conditional_probabilities_of_children(self, node_idx):
        """
        :param node_idx: node index
        :return: array of conditional probabilities of the children of a given node
        """
        prob_node_idx = self.probability_of_node(node_idx)
        children = self.children_of(node_idx)
        prob_children = self.__probability[children]
        return prob_children / prob_node_idx

    def __str__(self):
        return f"Scenario Tree\n+ Nodes: {self.num_nodes}\n+ Stages: {self.num_stages}\n" \
               f"+ Scenarios: {len(self.nodes_at_stage(self.num_stages - 1))}\n" \
               f"+ Data: {self.__data is not None}"

    def __repr__(self):
        return f"Scenario tree with {self.num_nodes} nodes, {self.num_stages} stages " \
               f"and {len(self.nodes_at_stage(self.num_stages - 1))} scenarios"

    @staticmethod
    def __circle_coord(rad, arc):
        return rad * np.cos(np.deg2rad(arc)), rad * np.sin(np.deg2rad(arc))

    @staticmethod
    def __goto_circle_coord(trt, rad, arc):
        trt.penup()
        trt.goto(ScenarioTree.__circle_coord(rad, arc))
        trt.pendown()

    @staticmethod
    def __draw_circle(trt, rad):
        trt.penup()
        trt.home()
        trt.goto(0, -rad)
        trt.pendown()
        trt.circle(rad)

    def __draw_leaf_nodes_on_circle(self, trt, radius, dot_size=6):
        trt.pencolor('gray')
        ScenarioTree.__draw_circle(trt, radius)
        leaf_nodes = self.nodes_at_stage(self.num_stages - 1)
        num_leaf_nodes = len(leaf_nodes)
        dv = 360 / num_leaf_nodes
        arcs = np.zeros(self.num_nodes)
        for i in range(num_leaf_nodes):
            ScenarioTree.__goto_circle_coord(trt, radius, i * dv)
            trt.pencolor('black')
            trt.dot(dot_size)
            trt.pencolor('gray')
            arcs[leaf_nodes[i]] = i * dv

        trt.pencolor('black')
        return arcs

    def __draw_nonleaf_nodes_on_circle(self, trt, radius, larger_radius, stage, arcs, dot_size=6):
        trt.pencolor('gray')
        ScenarioTree.__draw_circle(trt, radius)
        nodes = self.nodes_at_stage(stage)
        for n in nodes:
            mean_arc = np.mean(arcs[self.children_of(n)])
            arcs[n] = mean_arc
            ScenarioTree.__goto_circle_coord(trt, radius, mean_arc)
            trt.pencolor('black')
            trt.dot(dot_size)
            for nc in self.children_of(n):
                current_pos = trt.pos()
                trt.goto(ScenarioTree.__circle_coord(larger_radius, arcs[nc]))
                trt.goto(current_pos)
            trt.pencolor('gray')
        return arcs

    def bulls_eye_plot(self, dot_size=5, radius=300, filename=None):
        """
        Bull's eye plot of scenario tree

        :param dot_size: size of node [default: 5]
        :param radius: radius of largest circle [default: 300]
        :param filename: name of file, with .eps extension, to save the plot [default: None]
        """
        wn = turtle.Screen()
        wn.tracer(0)
        t = turtle.Turtle(visible=False)
        t.speed(0)

        arcs = self.__draw_leaf_nodes_on_circle(t, radius, dot_size)
        radius_step = radius / (self.num_stages - 1)
        for n in range(self.num_stages - 2, -1, -1):
            radius -= radius_step
            arcs = self.__draw_nonleaf_nodes_on_circle(
                t, radius, radius + radius_step, n, arcs, dot_size)

        wn.update()

        if filename is not None:
            wn.getcanvas().postscript(file=filename)
        wn.mainloop()


class MarkovChainScenarioTreeFactory:
    """
    Factory class to construct scenario trees from stopped Markov chains
    """

    def __init__(self, transition_prob, initial_distribution, num_stages, stopping_time=None):
        """
        :param transition_prob: transition matrix of the Markov chain
        :param initial_distribution: initial distribution of `w`
        :param num_stages: total number of stages or horizon of the scenario tree
        :param stopping_time: stopping time, which must be no larger than the number of stages [default: None]
        """
        self.__factory_type = "MarkovChain"
        if stopping_time is None:
            stopping_time = num_stages
        else:
            _check_stopping_time(num_stages, stopping_time)
        self.__transition_prob = transition_prob
        self.__initial_distribution = initial_distribution
        self.__num_stages = num_stages
        self.__stopping_time = stopping_time
        # --- check correctness of `transition_prob` and `initial_distribution`
        for pi in transition_prob:
            _check_probability_vector(pi)
        _check_probability_vector(initial_distribution)

    def __cover(self, i):
        pi = self.__transition_prob[i, :]
        return np.flatnonzero(pi)

    def __make_ancestors_values_stages(self):
        """
        :return: ancestors, values of w and stages
        """
        num_nonzero_init_distr = len(
            list(filter(lambda x: (x > 0), self.__initial_distribution)))
        # Initialise `ancestors`
        ancestors = np.zeros((num_nonzero_init_distr + 1,), dtype=int)
        ancestors[0] = -1  # node 0 does not have an ancestor
        # Initialise `values`
        values = np.zeros((num_nonzero_init_distr + 1,), dtype=int)
        values[0] = -1
        values[1:] = np.flatnonzero(self.__initial_distribution)
        # Initialise `stages`
        stages = np.ones((num_nonzero_init_distr + 1,), dtype=int)
        stages[0] = 0

        cursor = 1
        num_nodes_at_stage = num_nonzero_init_distr
        for stage_idx in range(1, self.__stopping_time):
            nodes_added_at_stage = 0
            cursor_new = cursor + num_nodes_at_stage
            for i in range(num_nodes_at_stage):
                node_id = cursor + i
                cover = self.__cover(values[node_id])
                length_cover = len(cover)
                ones = np.ones((length_cover,), dtype=int)
                ancestors = np.concatenate((ancestors, node_id * ones))
                nodes_added_at_stage += length_cover
                values = np.concatenate((values, cover))

            num_nodes_at_stage = nodes_added_at_stage
            cursor = cursor_new
            ones = np.ones(nodes_added_at_stage, dtype=int)
            stages = np.concatenate((stages, (1 + stage_idx) * ones))

        for stage_idx in range(self.__stopping_time, self.__num_stages):
            ancestors = np.concatenate(
                (ancestors, range(cursor, cursor + num_nodes_at_stage)))
            cursor += num_nodes_at_stage
            ones = np.ones((num_nodes_at_stage,), dtype=int)
            stages = np.concatenate((stages, (1 + stage_idx) * ones))
            values = np.concatenate((values, values[-num_nodes_at_stage::]))

        return ancestors, values, stages

    def __make_probability_values(self, ancestors, values, stages):
        """
        :return: probability
        """
        num_nonzero_init_distr = len(
            list(filter(lambda x: (x > 0), self.__initial_distribution)))
        # Initialise `probs`
        probs = np.zeros((num_nonzero_init_distr + 1,))
        probs[0] = 1
        probs[1:] = self.__initial_distribution[np.flatnonzero(
            self.__initial_distribution)]
        num_nodes = len(values)
        index = 0
        for i in range(num_nonzero_init_distr + 1, num_nodes):
            if stages[i] == self.__stopping_time + 1:
                index = i
                for j in range(index, num_nodes):
                    probs_new = probs[ancestors[j]]
                    probs = np.concatenate((probs, [probs_new]))

                break
            probs_new = probs[ancestors[i]] * \
                self.__transition_prob[values[ancestors[i]], values[i]]
            probs = np.concatenate((probs, [probs_new]))

        return probs

    def create(self):
        """
        Creates a scenario tree from the given Markov chain
        """
        # check input data
        ancestors, values, stages = self.__make_ancestors_values_stages()
        probs = self.__make_probability_values(ancestors, values, stages)
        tree = ScenarioTree(stages, ancestors, probs,
                            values, is_markovian=True)
        return tree
