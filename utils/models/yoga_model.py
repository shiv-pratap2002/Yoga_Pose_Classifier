from typing import List

import numpy as np

from utils.models.frame_model import FrameModel


class YogaVidModel(object):
    def __init__(self, pose_list: List[List[float]]):
        """
        Params
            pose_list: List of all landmarks for each frame of a video
        Args
            has_pose: bool; True if x pose is detected in the video, otherwise False
            xh_embedding: ndarray; Array of shape (n_frame, nb_connections * nb_connections)
        """
        self.has_pose = np.sum(pose_list) != 0
        self.pose_embedding = self._get_embedding_from_landmark_list(pose_list)

    @staticmethod
    def _get_embedding_from_landmark_list(
        POSE_list: List[List[float]]
    ) -> List[List[float]]:
        """
        Params
            pose_list: List of all landmarks for each frame of a video
        Return
            Array of shape (n_frame, nb_connections * nb_connections) containing
            the feature_vectors of the hand for each frame
        """
        embedding = []
        for frame_idx in range(len(POSE_list)):
            if np.sum(POSE_list[frame_idx]) == 0:
                continue

            pose_gesture = FrameModel(POSE_list[frame_idx])
            embedding.append(pose_gesture.feature_vector)
        return embedding
