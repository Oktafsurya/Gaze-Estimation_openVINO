# Computer Pointer Controller

In this project, we will be using the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly.
We will use 4 pre-trained model from openVINO:
1. [face-detection-adas-binary-0001](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
2. [head-pose-estimation-adas-0001](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
3. [facial-landmarks-35-adas-0002](https://docs.openvinotoolkit.org/latest/omz_models_intel_facial_landmarks_35_adas_0002_description_facial_landmarks_35_adas_0002.html)
4. [gaze-estimation-adas-0002](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

### The Pipeline
The flow of data will look like this:
<p align="center"> 
<img src=https://github.com/Oktafsurya/Gaze-Estimation_openVINO/blob/master/bin/pipeline.png>
</p>


## Project Set Up and Installation
- [Install the openVINO toolkit](https://docs.openvinotoolkit.org/latest/) according to the system being used. 
- Extract the submission file or clone the repository: 

  `https://github.com/Oktafsurya/Gaze-Estimation_openVINO.git`
  
- [Setup the virtual environment](https://docs.python.org/3.8/library/venv.html) in your system
- Install all the requirements

  `pip install -r requirements.txt`
  
- Download 4 pre-trained model from openVINO 

  `python src/model_downloader.py`

## Demo
Using CPU:

`python3 src/main.py --i bin/demo.mp4 --out output/result_FP16-INT8.avi --model_precision FP16-INT8`

Result:
<p align="center"> 
<img src=https://github.com/Oktafsurya/Gaze-Estimation_openVINO/blob/master/bin/gaze_estimation_ok.png>
</p>

or you can refer to the video result for each model precision
- [FP16](https://github.com/Oktafsurya/Gaze-Estimation_openVINO/blob/master/output/result_FP16.avi) (all model using FP16 precision)
- [FP16-INT8](https://github.com/Oktafsurya/Gaze-Estimation_openVINO/blob/master/output/result_FP16-INT8.avi) (all model using FP16-INT8 precision)
- [FP32](https://github.com/Oktafsurya/Gaze-Estimation_openVINO/blob/master/output/result_FP32.avi) (all model using FP32 precision)

## Documentation
Command line arguments needed by `model_downloader.py`

Argument|Type|Description
| ------------- | ------------- | -------------
--fd | Required (with default value) | face detection model that want to download.
--fl | Required (with default value) | facial landmark detection model that want to download.
--hp | Required (with default value) | head pose estimation model that want to download.
--ge | Required (with default value) | gaze estimation model that want to download.

Command line arguments needed by `main.py`

Argument|Type|Description
| ------------- | ------------- | -------------
--face_detection | Required (with default value) | Path to a face detection model xml file with a trained model.
--facial_landmark | Required (with default value) | Path to a facial landmark detection model xml file with a trained model.
--head_pose | Required (with default value) | Path to a head pose estimation model xml file with a trained model.
--gaze_model | Required (with default value) | Path to a gaze estimation model xml file with a trained model.
--in | Required | Path to image or video file or CAM.
--out | Required | path to output video.

## Benchmarks
Benchmarking is done using a laptop with specifications:
- Brand     : ASUS
- CPU       : Intel® Core™ i7-4720HQ CPU @ 2.60GHz × 8 
- Graphics  : GeForce GTX 950M/PCIe/SSE2
- RAM       : 8 GB
- OS        : Ubuntu 16.04 LTS 64-bit

#### CPU

| Properties            | FP16        | FP16-INT8   | FP32        |
| ----------------------| ----------- | ----------- | ----------- |
|Total Model Loading    | 707.41ms    | 2216.518ms  | 690.83ms    |
|Total Inference Time   | 74.1s       | 74.2s       | 74.0s       |
|FPS                    | 0.796fps    | 0.795fps    | 0.797fps    |

| Loading time each model (in ms)   |  FP16       | FP16-INT8   | FP32        |
| ----------------------            | ----------- | ----------- | ----------- |
|Face detection                     | *           | *           | *           |
|Facial landmark                    | 397.866     | 1728.518    | 348.82      |
|Head pose                          | 75.923      | 192.77      | 61.65       |
|Gaze estimation                    | 91.259      | 166.07      | 75.10       |

## Results
From the benchmarking result above, we can conclude that model with lower precision give us faster total inference time, total time to load all model and also fps. Model with higher precision for example FP16-INT8 tend to give slower total inference time, total time to load all model and fps.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.
