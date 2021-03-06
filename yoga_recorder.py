import pandas as pd
import numpy as np
from collections import Counter

from utils.dtw import dtw_distances
from utils.models.yoga_model import YogaVidModel
from utils.landmark_utils import extract_landmarks


class YogaRecorder(object):
    def __init__(self, reference_pose: pd.DataFrame, seq_len=50):
        # Variables for recording
        self.is_recording = False
        self.seq_len = seq_len

        # List of results stored each frame
        self.recorded_results = []

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_pose = reference_pose

    def record(self):
        """
        Initialize sign_distances & start recording
        """
        self.reference_pose["distance"].values[:] = 0
        self.is_recording = True

    def process_results(self, results) -> (str, bool):
        """
        If the SignRecorder is in the recording state:
            it stores the landmarks during seq_len frames and then computes the sign distances
        :param results: mediapipe output
        :return: Return the word predicted (blank text if there is no distances)
                & the recording state
        """
        if self.is_recording:
            if len(self.recorded_results) < self.seq_len:
                self.recorded_results.append(results)
            else:
                self.compute_distances()
                print(self.reference_pose)

        if np.sum(self.reference_pose["distance"].values) == 0:
            return "", self.is_recording
        return self._get_sign_predicted(), self.is_recording

    def compute_distances(self):
        """
        Updates the distance column of the reference_signs
        and resets recording variables
        """
        pose_list = []
        for results in self.recorded_results:
            pose = extract_landmarks(results)
            pose_list.append(pose)
            # right_hand_list.append(right_hand)

        # Create a SignModel object with the landmarks gathered during recording
        recorded_pose = YogaVidModel(pose_list)

        # Compute sign similarity with DTW (ascending order)
        self.reference_pose = dtw_distances(recorded_pose, self.reference_pose)

        # Reset variables
        self.recorded_results = []
        self.is_recording = False

    def _get_sign_predicted(self, batch_size=5, threshold=0.5):
        """
        Method that outputs the yoga pose that appears the most in the list of closest
        reference yoga poses, only if its proportion within the batch is greater than the threshold

        :param batch_size: Size of the batch of reference yoga that will be compared to the recorded sign
        :param threshold: If the proportion of the most represented sign in the batch is greater than threshold,
                        we output the sign_name
                          If not,
                        we output "Sign not found"
        :return: The name of the predicted sign
        """
        # Get the list (of size batch_size) of the most similar reference signs
        yoga_names = self.reference_pose.iloc[:batch_size]["name"].values

        # Count the occurrences of each sign and sort them by descending order
        yoga_counter = Counter(yoga_names).most_common()

        predicted_sign, count = yoga_counter[0]
        # if count / batch_size < threshold:
        #     return "Signe inconnu"
        return predicted_sign
