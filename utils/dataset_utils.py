import os

import pandas as pd
from tqdm import tqdm

from utils.models.yoga_model import YogaVidModel
from utils.landmark_utils import save_landmarks_from_video, load_array


def load_dataset():
    videos = [
        file_name.replace(".mp4", "")
        for root, dirs, files in os.walk(os.path.join("data", "videos"))
        for file_name in files
        if file_name.endswith(".mp4")
    ]#List of all videos in data/videos file
    dataset = [
        file_name.replace(".pickle", "").replace("pose_", "")
        for root, dirs, files in os.walk(os.path.join("data", "dataset"))
        for file_name in files
        if file_name.endswith(".pickle") and file_name.startswith("pose_")
    ]#Landmarks data for given video name list
    # Create the dataset from the reference videos
    videos_not_in_dataset = list(set(videos).difference(set(dataset)))
    n = len(videos_not_in_dataset)
    if n > 0:#To add new videos
        print(f"\nExtracting landmarks from new videos: {n} videos detected\n")

        for idx in tqdm(range(n)):
            save_landmarks_from_video(videos_not_in_dataset[idx])

    return videos


def load_reference_poses(videos):
    #List of video names: Input
    # Returns data frame with data about reference poses
    reference_poses = pd.DataFrame(columns=["name", "yoga_pose", "distance"])
    for video_name in videos:
        yoga_name = video_name.split("-")[0]
        path = os.path.join("data", "dataset", yoga_name, video_name)
        
        pose_list = load_array(os.path.join(path, f"pose_{video_name}.pickle"))
        # print(pose_list)
        #Loads np.array from given path
        reference_poses = reference_poses.append(
            {
                "name": yoga_name,
                "yoga_pose": YogaVidModel(pose_list),
                "distance": 0,
            },
            ignore_index=True,
        )
    print(
        f'Dictionary count: {reference_poses[["name", "yoga_pose"]].groupby(["name"]).count()}'
    )
    # reference_signs.to_csv('ref_signs.csv', encoding='utf-8')
    return reference_poses
