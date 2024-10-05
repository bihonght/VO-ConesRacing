import numpy as np

class MapPoint:
    factory_id_ = 0 # or 1

    def __init__(self, pos, descriptor, norm, r=0, g=0, b=0):
        """
        Initializes a MapPoint instance.

        Args:
            pos (np.ndarray or tuple): The 3D position of the map point (x, y, z).
            descriptor (np.ndarray): Descriptor for matching (e.g., ORB feature).
            norm (np.ndarray): Vector pointing from the camera center to the point.
            r (int): Red color component (default 0).
            g (int): Green color component (default 0).
            b (int): Blue color component (default 0).
        """
        self.id_ = MapPoint.factory_id_
        MapPoint.factory_id_ += 1

        self.pos_ = np.array(pos, dtype=np.float32)
        self.norm_ = np.array(norm, dtype=np.float32)
        self.color_ = [r, g, b]  # RGB color components
        self.descriptor_ = descriptor

        # Properties for local mapping
        self.good_ = True
        self.matched_times_ = 0  # Being an inlier in pose estimation
        self.visible_times_ = 0  # Being visible in the current frame

    def set_pos(self, pos):
        """
        Sets the 3D position of the map point.

        Args:
            pos (np.ndarray or tuple): The new 3D position (x, y, z).
        """
        self.pos_ = np.array(pos, dtype=np.float32)

