from math import sqrt

import numpy as np

from cvkit import MAGIC_NUMBER


class Part(np.ndarray):

    def __new__(cls, arr, name, likelihood):
        """
        Represents a body part of the tracked subject.
        :param arr: Array of N values defining the position in N-dimensional space
        :param_type arr: list,:class:'numpy.ndarray'
        :param name: Name of the body part
        :param_type name: string
        :param likelihood: A value indicating confidence in the accuracy of the position defined by arr
        :param_type likelihood: float
        """
        obj = np.asarray(arr).view(cls)
        assert obj.ndim == 1
        obj.name = name
        obj.likelihood = likelihood
        return obj

    def distance(self, obj):
        assert len(obj) == len(self)
        """
        Computes distance between two Parts.
        :param obj: N-dimensional vector
        :param_type obj: :class:'cvkit.pose_estimation.Part',:class:'numpy.ndarray'
        :return: Distance between the part and the target vector
        :rtype: float
        """
        return sqrt(np.sum(np.square(np.subtract(obj, self))))

    def magnitude(self):
        """
        Computes the magnitude of the Part.
        :return: Magnitude of the Part.
        :rtype: float
        """
        return sqrt(np.sum(np.square(self)))

    def __lt__(self, other: float):
        """
        Checks if the likelihood of the Part is lesser than a certain value
        :param other: A number for comparison
        :param_type other: float
        :return: Whether the likelihood is lesser than the value.
        :rtype: bool
        """
        return self.likelihood < other

    def __gt__(self, other: float):
        """
        Checks if the likelihood of the Part is greater than a certain value
        :param other: A number for comparison
        :param_type other: float
        :return: Whether the likelihood is greater than the value.
        :rtype: bool
        """
        return self.likelihood > other

    def __ge__(self, other: float):
        """
        Checks if the likelihood of the Part is greater than or equal to a certain value
        :param other: A number for comparison
        :param_type other: float
        :return: Whether the likelihood is greater than or equal to the value.
        :rtype: bool
        """
        return self.likelihood >= other

    def __le__(self, other: float):
        """
        Checks if the likelihood of the Part is lesser than or equal to a certain value
        :param other: A number for comparison
        :param_type other: float
        :return: Whether the likelihood is lesser than or equal to the value.
        :rtype: bool
        """
        return self.likelihood <= other

    def __add__(self, other):
        """
        Adds an N-dimensional vector or a scalar and creates a new Part Object
        :param other: N-dimensional vector
        :param_type other: :class:`numpy.ndarray', list
        :return: A new Part created from resulting vector.
        :rtype: :class:`cvkit.pose_estimation.Part'
        """
        return Part(super().__add__(other), self.name, self.likelihood)

    def __sub__(self, other):
        """
        Subtracts an N-dimensional vector or a scalar and creates a new Part Object
        :param other: N-dimensional vector
        :param_type other: :class:`numpy.ndarray', list
        :return: A new Part created from resulting vector.
        :rtype: :class:`cvkit.pose_estimation.Part'
        """
        return Part(super().__sub__(other), self.name, self.likelihood)

    def __mul__(self, other):
        """
        Multiplies an N-dimensional vector or a scalar and creates a new Part Object
        :param other: N-dimensional vector
        :param_type other: :class:`numpy.ndarray', list, float
        :return: A new Part created from resulting vector.
        :rtype: :class:`cvkit.pose_estimation.Part'
        """
        return Part(super().__mul__(other), self.name, self.likelihood)

    def __radd__(self, other):
        """
        Adds an N-dimensional vector or a scalar and creates a new Part Object
        :param other: N-dimensional vector
        :param_type other: :class:`numpy.ndarray', list, float
        :return: A new Part created from resulting vector.
        :rtype: :class:`cvkit.pose_estimation.Part'
        """
        return Part(super().__radd__(other), self.name, self.likelihood)

    def numpy(self):
        """
        Creates a numpy array from Part
        :return: An N-Dimensional numpy array
        :rtype: :class:'numpy.ndarray'
        """
        return np.array(self)


class Skeleton:

    def __init__(self, body_parts: list, part_map: dict = None, likelihood_map: dict = None, behaviour=[], dims=3):
        """

        :param body_parts:
        :param part_map:
        :param likelihood_map:
        :param behaviour:
        :param dims:
        """
        self.body_parts = body_parts
        self.body_parts_map = {}
        candidates = body_parts.copy()
        self.behaviour = behaviour
        self.dims = dims
        for name in part_map.keys():
            candidates.remove(name)
            self.body_parts_map[name] = Part(part_map[name], name, likelihood_map[name])
            try:
                assert self.body_parts_map[name].shape == (dims,)
            except:
                pass
        for name in candidates:
            self.body_parts_map[name] = Part([MAGIC_NUMBER] * dims, name, .0)

    def __getitem__(self, item):
        return self.body_parts_map[item] if item in self.body_parts_map.keys() else None

    def __setitem__(self, name, val):
        self.body_parts_map[name] = val

    def __str__(self):
        ret = ""
        for p in self.body_parts_map:
            ret += f"{p:10}: {str(self[p])} ({np.round(self.body_parts_map[p].likelihood, 2)})\n"
        return ret

    def __rsub__(self, other):
        val = {}
        prob = {}
        for name in self.body_parts_map:
            val[name] = (other[name] - self[name]) if type(other) == Skeleton else other - self[name]
            if type(other) == Skeleton:
                prob[name] = min(self[name].likelihood, other[name].likelihood)
        return Skeleton(list(self.body_parts_map.keys()), None, val, prob)

    def __sub__(self, other):
        val = {}
        prob = {}
        for name in self.body_parts_map:
            val[name] = (self[name] - other[name]) if type(other) == Skeleton else self[name] - other
            if type(other) == Skeleton:
                prob[name] = min(self[name].likelihood, other[name].likelihood)
        return Skeleton(list(self.body_parts_map.keys()), None, val, prob)

    def __radd__(self, other):
        val = {}
        prob = {}
        for name in self.body_parts_map:
            val[name] = (self[name] + other[name]) if type(other) == Skeleton else self[name] + other
            if type(other) == Skeleton:
                prob[name] = min(self[name].likelihood, other[name].likelihood)
        return Skeleton(list(self.body_parts_map.keys()), None, val, prob)

    def __add__(self, other):
        val = {}
        prob = {}
        for name in self.body_parts_map:
            val[name] = (self[name] + other[name]) if type(other) == Skeleton else self[name] + other
            if type(other) == Skeleton:
                prob[name] = min(self[name].likelihood, other[name].likelihood)
        return Skeleton(list(self.body_parts_map.keys()), None, val, prob)

    def __mul__(self, other):
        val = {}
        prob = {}
        for name in self.body_parts_map:
            val[name] = (self[name] * other[name]) if type(other) == Skeleton else self[name] * other
            if type(other) == Skeleton:
                prob[name] = min(self[name].likelihood, other[name].likelihood)
        return Skeleton(list(self.body_parts_map.keys()), None, val, prob)

    def __rmul__(self, other):
        val = {}
        prob = {}
        for name in self.body_parts_map:
            val[name] = (self[name] * other[name]) if type(other) == Skeleton else self[name] * other
            if type(other) == Skeleton:
                prob[name] = min(self[name].likelihood, other[name].likelihood)
        return Skeleton(list(self.body_parts_map.keys()), None, val, prob)

    def __eq__(self, other):
        try:
            return all([np.all(self[part] == other[part]) for part in self.body_parts])
        except KeyError:
            return False

    def __iter__(self):
        for part in self.body_parts_map.values():
            yield part

    def __len__(self):
        return len(self.body_parts)

    def normalize(self, max_lim: np.ndarray, min_lim: np.ndarray):
        max_lim = np.array(max_lim)
        min_lim = np.array(min_lim)
        if max_lim.shape != min_lim.shape != (self.dims,):
            raise Exception(f"Maximum limit and Minimum limit should have ({self.dims},) shape")
        normalize_fn = lambda x: (x - min_lim) / (max_lim - min_lim)
        part_map = {}
        likelihood_map = {}
        for part in self.body_parts_map.keys():
            part_map[part] = normalize_fn(self.body_parts_map[part])
            likelihood_map[part] = self.body_parts_map[part].likelihood
        return Skeleton(self.body_parts, part_map, likelihood_map, self.behaviour)
