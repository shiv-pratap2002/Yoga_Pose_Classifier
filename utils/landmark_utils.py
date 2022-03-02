import cv2
import os
import numpy as np
import pickle as pkl
import mediapipe as mp
from utils.mediapipe_utils import mediapipe_detection


def landmark_to_array(mp_landmark_list):
    """Inputs a list to return a np array of size (nb_keypoints x 3)"""
    keypoints = []
    for landmark in mp_landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)


def extract_landmarks(results):
    """Extract the results of pose and convert into np array
    if a pose doesn't appear, return an array of zeros
    :param results: mediapipe object that contains the 3D position of all keypoints
    :return: Two np arrays of size (1, 33 * 3) = (1, nb_keypoints * nb_coordinates) corresponding to both hands
    """
    pose = landmark_to_array(results.pose_landmarks).reshape(99).tolist()
    return pose


def save_landmarks_from_video(video_name):
    
    landmark_list = {"pose": []}
    yoga_name = video_name.split("-")[0]
    # Set the Video stream
    cap = cv2.VideoCapture(
        os.path.join("data", "videos", yoga_name,video_name + ".mp4")
    )
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        # print(cap.isOpened())#This is outputting false
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # Store results
                pose = extract_landmarks(results)
                landmark_list["pose"].append(pose)
            else:
                break

        cap.release()
    # Create the folder of the pose if it doesn't exists
    path = os.path.join("data", "dataset", yoga_name)
    if not os.path.exists(path):
        os.mkdir(path)

    # Create the folder of the video data if it doesn't exists
    data_path = os.path.join(path, video_name)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Saving the landmark_list in the correct folder
    save_array(
        landmark_list["pose"], os.path.join(data_path, f"pose_{video_name}.pickle")
    )

def save_array(arr, path):
    file = open(path, "wb")
    pkl.dump(arr, file)
    file.close()


def load_array(path):
    file = open(path, "rb")
    arr = pkl.load(file)
    file.close()
    return np.array(arr)
