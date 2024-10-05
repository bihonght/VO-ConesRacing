class Map:
    def __init__(self):
        # Initialize keyframes and map points as dictionaries
        self.keyframes_ = {}  # Stores keyframes with their id as the key
        self.map_points_ = {}  # Stores map points with their id as the key

    def insert_key_frame(self, State):
        """
        Inserts a keyframe into the map. If the frame already exists, update it.

        Args:
            frame (Frame): The keyframe to insert.
        """
        if State.frame_id not in self.keyframes_:
            self.keyframes_[State.frame_id] = State
        else:
            self.keyframes_[State.frame_id] = State
        print(f"Insert keyframe!!! frame_id = {State.frame_id}, total keyframes = {len(self.keyframes_)}")

    def insert_map_point(self, map_point):
        """
        Inserts a map point into the map. If the map point already exists, update it.

        Args:
            map_point (MapPoint): The map point to insert.
        """
        if map_point.id_ not in self.map_points_:
            self.map_points_[map_point.id_] = map_point
        else:
            self.map_points_[map_point.id_] = map_point

    def find_key_frame(self, frame_id):
        """
        Finds a keyframe by its ID.

        Args:
            frame_id (int): The ID of the frame to find.

        Returns:
            Frame: The found frame, or None if not found.
        """
        return self.keyframes_.get(frame_id, None)

    def has_key_frame(self, frame_id):
        """
        Checks if a keyframe exists in the map.

        Args:
            frame_id (int): The ID of the frame to check.

        Returns:
            bool: True if the keyframe exists, False otherwise.
        """
        return frame_id in self.keyframes_
