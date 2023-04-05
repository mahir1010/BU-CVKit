from math import sqrt

import numpy as np

from cvkit import MAGIC_NUMBER


class Part(np.ndarray):

    def __new__(cls, arr, name, likelihood):
        obj = np.asarray(arr).view(cls)
        obj.name = name
        obj.likelihood = likelihood
        return obj

    def distance(self, obj):
        return sqrt(np.sum(np.square(np.subtract(obj, self))))

    def magnitude(self):
        return sqrt(np.sum(np.square(self)))

    def __lt__(self, other: float):
        return self.likelihood < other

    def __gt__(self, other: float):
        return self.likelihood > other

    def __ge__(self, other: float):
        return self.likelihood >= other

    def __le__(self, other: float):
        return self.likelihood <= other

    def __add__(self, other):
        return Part(super().__add__(other), self.name, self.likelihood)

    def __sub__(self, other):
        return Part(super().__sub__(other), self.name, self.likelihood)

    def __mul__(self, other):
        return Part(super().__mul__(other), self.name, self.likelihood)

    def __radd__(self, other):
        return Part(super().__radd__(other), self.name, self.likelihood)

    def numpy(self):
        return np.array(self)


class Skeleton:
    def __init__(self, body_parts: list, part_map: dict = None, likelihood_map: dict = None, behaviour=[], dims=3):
        self.body_parts = body_parts
        self.body_parts_map = {}
        candidates = body_parts.copy()
        self.behaviour = behaviour
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
        if max_lim.shape != min_lim.shape != (3,):
            raise Exception("Maximum limit and Minimum limit should have (3,) shape")
        normalize_fn = lambda x: (x - min_lim) / (max_lim - min_lim)
        part_map = {}
        likelihood_map = {}
        for part in self.body_parts_map.keys():
            part_map[part] = normalize_fn(self.body_parts_map[part])
            likelihood_map[part] = self.body_parts_map[part].likelihood
        return Skeleton(self.body_parts, part_map, likelihood_map, self.behaviour)
