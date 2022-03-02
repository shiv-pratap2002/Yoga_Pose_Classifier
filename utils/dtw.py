import pandas as pd
from fastdtw import fastdtw
import numpy as np
from utils.models.yoga_model import YogaVidModel


def dtw_distances(recorded_yoga: YogaVidModel, reference_signs: pd.DataFrame):
    """
    Use DTW to compute similarity between the recorded sign & the reference signs

    :param recorded_sign: a YogaVidModel object containing the data gathered during record
    :param reference_signs: pd.DataFrame
                            columns : name, dtype: str
                                      sign_model, dtype: SignModel
                                      distance, dtype: float64
    :return: Return a pose dictionary sorted by the distances from the recorded sign
    """
    # Embeddings of the recorded sign
    rec_pose= recorded_yoga.pose_embedding

    for idx, row in reference_signs.iterrows():
        # Initialize the row variables
        ref_yoga_name, ref_yoga_model, _ = row

        # If the reference sign has the same number of hands compute fastdtw
        if (recorded_yoga.has_pose == ref_yoga_model.has_pose):
            ref_pose = ref_yoga_model.pose_embedding

            if recorded_yoga.has_pose:
                row["distance"] += list(fastdtw(rec_pose, ref_pose))[0]

        # If not, distance equals infinity
        else:
            row["distance"] = np.inf
    return reference_signs.sort_values(by=["distance"])
