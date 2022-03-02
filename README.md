
# Yoga_Pose_Classifier - using MediaPipe and DTW

This project is an implementation of a Yoga Pose Classifier using the **MediaPipe** library 
for landmark extraction and **Dynamic Time Warping** (DTW) as a similarity metric between yoga poses.

___

## Set up

### 1. Open terminal and go to the Project directory

### 2. Install the necessary libraries

- ` pip install -r requirements.txt `

### 3. Import Videos of signs which will be considered as reference
The architecture of the `videos/` folder must be:
```
|data/
    |-videos/
          |-Tadasan/
            |-<video_of_tadasan_1>.mp4
            |-<video_of_tadasan_2>.mp4
            ...
          |-Bhujasan/
            |-<video_of_bhujasan_1>.mp4
            |-<video_of_bhujasan_2>.mp4
            ...
```

Data Set Used : https://archive.org/download/YogaVidCollected
> N.B. The current dataset is insufficient to obtain good results. Feel free to add more links or import your own videos 

### 4. Load the dataset and turn on the Webcam

- ` python main.py `

### 5. Press the "r" key to record the sign. 

___
## Code Description

### *Landmark extraction (MediaPipe)*

- The **Holistic Model** of MediaPipe allows us to extract the keypoints of the Hands, Pose and Face models.
For now, the implementation only uses the Pose model to predict the yoga pose.


### *Frame Model*

- In this project a **FrameModel** has been created to define the pose at each frame. 
If person making is not present we set all the positions to zero.

- In order to be **invariant to orientation and scale**, the **feature vector** of the
HandModel is a **list of the angles** between all the connexions of the hand.

### *Yoga Model*

- The **YogaVidModel** is created from a list of landmarks (extracted from a video)

- For each frame, we **store** the **feature vectors** for pose.

### *Yoga Recorder*

- The **YogaRecorder** class **stores** the FrameModels of poses for each frame **when recording**.
- Once the recording is finished, it **computes the DTW** of the recorded yoga pose and 
all the reference yoga poses present in the dataset.
- Finally, a voting logic is added to output a result only if the prediction **confidence** is **higher than a threshold**.

### *Dynamic Time Warping*

-  DTW is widely used for computing time series similarity.

- In this project, we compute the DTW of the variation of hand connexion angles over time.

___

## References

 - [Pham Chinh Huu, Le Quoc Khanh, Le Thanh Ha : Human Action Recognition Using Dynamic Time Warping and Voting Algorithm](https://www.researchgate.net/publication/290440452)
 - [Mediapipe : Pose classification](https://google.github.io/mediapipe/solutions/pose_classification.html)
