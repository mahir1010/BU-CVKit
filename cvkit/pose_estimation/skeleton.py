from math import sqrt

import numpy as np

from cvkit import MAGIC_NUMBER


class Part(np.ndarray):
    """
    Represents a body part of the tracked subject.

    .. highlight:: python
    .. code-block:: python

        #2D Part pointing to l_eye with 0.7 likelihood.
        part = Part([100,200],'l_eye',0.7)
        #3D Part pointing to l_eye with 0.5 likelihood.
        part_l_eye_3d = Part([100,100,50],'l_eye',0.5)
        #3D Part pointing to r_eye with 0.5 likelihood.
        part_r_eye_3d = Part([100,100,50],'r_eye',0.5)
        #3D Part pointing to eye_mid with 0.5 likelihood.
        part_eye_mid_3d = (part_l_eye_3d + part_r_eye_3d)/2

    :param arr: Array of N values defining the position in N-dimensional space
    :param_type arr: list,:class:'numpy.ndarray'
    :param name: Name of the body part
    :param_type name: string
    :param likelihood: A value indicating confidence in the accuracy of the position defined by arr
    :param_type likelihood: float
    """
    def __new__(cls, arr, name, likelihood):

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
        Checks if the likelihood of the Part is lesser than a certain value.


        :param other: A number for comparison
        :param_type other: float
        :return: Whether the likelihood is lesser than the value.
        :rtype: bool
        """
        return self.likelihood < other

    def __gt__(self, other: float):
        """
        Checks if the likelihood of the Part is greater than a certain value.


        :param other: A number for comparison
        :param_type other: float
        :return: Whether the likelihood is greater than the value.
        :rtype: bool
        """
        return self.likelihood > other

    def __ge__(self, other: float):
        """
        Checks if the likelihood of the Part is greater than or equal to a certain value.


        :param other: A number for comparison
        :param_type other: float
        :return: Whether the likelihood is greater than or equal to the value.
        :rtype: bool
        """
        return self.likelihood >= other

    def __le__(self, other: float):
        """
        Checks if the likelihood of the Part is lesser than or equal to a certain value.


        :param other: A number for comparison
        :param_type other: float
        :return: Whether the likelihood is lesser than or equal to the value.
        :rtype: bool
        """
        return self.likelihood <= other

    def __add__(self, other):
        """
        Adds an N-dimensional vector or a scalar and creates a new Part Object.


        :param other: N-dimensional vector
        :param_type other: :class:`numpy.ndarray', list
        :return: A new Part created from resulting vector.
        :rtype: :class:`cvkit.pose_estimation.Part'
        """
        return Part(super().__add__(other), self.name, self.likelihood)

    def __sub__(self, other):
        """
        Subtracts an N-dimensional vector or a scalar and creates a new Part Object.


        :param other: N-dimensional vector
        :param_type other: :class:`numpy.ndarray', list
        :return: A new Part created from resulting vector.
        :rtype: :class:`cvkit.pose_estimation.Part'
        """
        return Part(super().__sub__(other), self.name, self.likelihood)

    def __mul__(self, other):
        """
        Multiplies an N-dimensional vector or a scalar and creates a new Part Object.


        :param other: N-dimensional vector
        :param_type other: :class:`numpy.ndarray', list, float
        :return: A new Part created from resulting vector.
        :rtype: :class:`cvkit.pose_estimation.Part'
        """
        return Part(super().__mul__(other), self.name, self.likelihood)

    def __radd__(self, other):
        """
        Adds an N-dimensional vector or a scalar and creates a new Part Object.


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
    """ This class represents the skeleton of the tracked subject.

    .. highlight:: python
    .. code-block:: python

        body_parts = ['snout','headBase']
        data_map_1 = {'snout':[200,300,50],'headBase':[200,270,100]}
        likelihood_map_1 = {'snout':0.7,'headBase':0.8}
        current_behaviours = ['rearing']

        # Skeleton at t = 0
        # (list of bodyparts, data dictionary, likelihood dictionary, behaviour list (default empty), dimensions (default 3)
        # For 2D skeleton set dims = 2
        skeleton_1 = Skeleton(body_parts,data_map_1,likelihood_map_1,current_behaviours)

        data_map_2 = {'snout':[100,300,50],'headBase':[100,270,100]}
        likelihood_map_2 = {'snout':0.7,'headBase':0.8}

        # Skeleton at t = 1
        skeleton_2 = Skeleton(body_parts,data_map_2,likelihood_map_2)

        # Displacement
        displacement = skeleton_2 - skeleton_1
        print(displacement['snout'],displacement['headBase'])

        #Head Direction
        head_direction = skeleton_1['snout'] - skeleton_1['headBase']

        #Support broadcast operations
        skeleton_1 = skeleton_1 + [10,20,-5]    # non-uniform translation
        skeleton_1 = skeleton_1 + 5             # uniform translation
        skeleton_1 = skeleton_1 * 2             # uniform scaling
        skeleton_1 = skeleton_1 * [0.5,1,0.5]   # non-uniform scaling

        #Supports elementwise operations
        skeleton_3 = skeleton_1 + skeleton_2
        skeleton_3 = skeleton_1 * skeleton_2

        #Normalize skeleton between 0 and 1.0
        min_coordinates = [0,0,0] # Define minimum coordinate values
        max_coordinates = [1000,1000,500] # Define maximum coordinate values
        skeleton_1 = skeleton_1.normalize(min_coordinates,max_coordinates)

    :param body_parts: list of body parts
    :type body_parts: list[str]
    :param part_map: A dictionary where the key is body part and value is its corresponding n-dimensional data.
    :type part_map: dict
    :param likelihood_map: A dictionary where the key is body part and value is its corresponding likelihood data.
    :type likelihood_map: dict
    :param behaviour: list of labels defining the behaviour of the subject at current frame.
    :type behaviour: list[str]
    :param dims: Dimension of underlying data.
    :type dims: int
    """

    def __init__(self, body_parts: list, part_map: dict = None, likelihood_map: dict = None, behaviour=[], dims=3):

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

    def normalize(self, max_lim, min_lim):
        """ Normalizes the skeleton so that the values range from 0.0 to 1.0

        :param max_lim: The maximum limit of the coordinate system. n-dimensional list of coordinates.
        :type max_lim: list[float]
        :param min_lim: The minimum limit of the coordinate system. n-dimensional list of coordinates.
        :type min_lim: list[float]
        :return: Normalized Skeleton Object
        :rtype: Skeleton
        """
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
