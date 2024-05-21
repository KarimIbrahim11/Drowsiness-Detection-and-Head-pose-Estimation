# Drowsiness-Detection-and-Head-pose-Estimation

Drowsiness detection and Head pose Estimation in realtime using Laptop's camera. The project uses python, openCV and mediapipe. 
![headpose](https://github.com/KarimIbrahim11/Drowsiness-Detection-and-Head-pose-Estimation/assets/47744559/933d31e2-d7ef-4f03-bb23-b1b8b77aa4a1)

## Demo 
Showcasing the feature
https://github.com/KarimIbrahim11/Drowsiness-Detection-and-Head-pose-Estimation/assets/47744559/59f9e0bb-690a-4ca4-b3cb-47d6f7386461

## Solution

1- Face Mesh detection using mediapipe 
2- The facial landmark points `"left": [362, 385, 387, 263, 373, 380]` & `"right": [33, 160, 158, 133, 153, 144]` were chosen to calculate the inter-eyed distance and the EAR. 
3- EAR fixed is 0.25 and the time is 2 Seconds to start reporting Drowsiness.
4- For Headpose estimation, I started off by calculating the rotation and translation vectors between 2d and 3d facial landmarks using `cv2.solvePnP()` function
5- the rotation matrix was then decomposed to find the angles x, y, z
6- Angles solely, were used statically in the code to identify the head orientation (nominal,up,down,right,left)
7- The angles were then used to display the 3 coordinate vectors. 

## Assumptions and Areas of improvement

- Focal Length, Camera matrix and Distortion Matrix were all assumed and were not calculated.
- Currently running 37 FPS. Room for improvement.

## Results duplication
1- create a virtual env, I personaly use conda. 
2- `conda env create -f environment.yaml`
3- `conda activate env`
4- `python main.py`

