import abc


class Refiner:

    def __init__(self):
        pass

    @abc.abstractmethod
    def refine(self, rgb, depth, intrinsics, obj_roi, obj_mask, obj_id,
               estimate, iterations):
        pass
