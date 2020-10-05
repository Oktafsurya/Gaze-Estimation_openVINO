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
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
