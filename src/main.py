from argparse import ArgumentParser
from input_feeder import InputFeeder
import os

from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FacialLandmarkDetection
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController
import cv2
import numpy as np
import logging
import time


def arg_parse():
    parser = ArgumentParser()

    parser.add_argument("--face_det", dest = 'face_detection', help ="Path to a face detection model xml file with a trained model.",
                        default="models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml")
    parser.add_argument("--head_pose", dest = 'head_pose', help ="Path to a head pose estimation model xml file with a trained model.",
                        default="models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml")
    parser.add_argument("--facial_land", dest = 'facial_landmark', help ="Path to a facial landmark detection model xml file with a trained model.",
                        default="models/intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml")
    parser.add_argument("--gaze_model", dest = 'gaze_model', help ="Path to a gaze estimation model xml file with a trained model.",
                        default="models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml")

    parser.add_argument("--in", dest = 'input', type=str, required=True,
                        help="Path to image or video file or CAM")

    parser.add_argument("--out", dest = 'output', help="path to output video")
    
    return parser.parse_args()

def face_detection_process(face_model, image):

    img = face_model.preprocess_input(image)
    outputs = face_model.sync_inference(img)
    outputs = outputs['detection_out']
    face_coordinates = face_model.preprocess_output(image, outputs)

    return face_coordinates

def head_pose_process(head_pose_model, image):

    img = head_pose_model.preprocess_input(image)
    outputs = head_pose_model.sync_inference(img)
    pitch, roll, yaw = head_pose_model.preprocess_output(outputs)

    return [pitch, roll, yaw]

def facial_landmark_process(facial_landmark_model, image):

    img = facial_landmark_model.preprocess_input(image)
    outputs = facial_landmark_model.sync_inference(img)
    facial_landmark, left_eye_coord, right_eye_coord = facial_landmark_model.preprocess_output(image, outputs)

    return facial_landmark, left_eye_coord, right_eye_coord

def gaze_estimation_process(gaze_estimation_model, head_pose, cropped_left_eye_image, cropped_right_eye_image):
    img_left_eye = gaze_estimation_model.preprocess_input(cropped_left_eye_image)
    img_right_eye = gaze_estimation_model.preprocess_input(cropped_right_eye_image)
    outputs = gaze_estimation_model.sync_inference(head_pose, img_left_eye, img_right_eye)
    gaze_angles = gaze_estimation_model.preprocess_output(outputs)

    return gaze_angles


if __name__ ==  '__main__':

    logger = logging.getLogger()
    args = arg_parse()
    input_file = args.input

    face_model_path = args.face_detection
    head_pose_path = args.head_pose
    facial_landmark_path = args.facial_landmark
    gaze_model_path = args.gaze_model

    face_model = FaceDetection(model_name=face_model_path)
    head_pose_model = HeadPoseEstimation(model_name=head_pose_path)
    facial_landmark_model = FacialLandmarkDetection(model_name=facial_landmark_path)
    gaze_estimation_model = GazeEstimation(model_name=gaze_model_path)

    mouse_controller = MouseController('medium', 'fast')

    start_time = time.time()
    face_model.load_model()
    face_loading_time = (time.time() - start_time)*1000

    head_start_time = time.time()
    head_pose_model.load_model()
    head_pose_time = (time.time()-head_start_time)*1000

    facial_landmark_start = time.time()
    facial_landmark_model.load_model()
    facial_landmark_time = (time.time() - facial_landmark_start)*1000

    gaze_model_start = time.time()
    gaze_estimation_model.load_model()
    gaze_model_time = (time.time() - gaze_model_start)*1000

    total_loading_time = (time.time() - start_time)*1000

    face_model.check_model()
    head_pose_model.check_model()
    facial_landmark_model.check_model()
    gaze_estimation_model.check_model()

    if input_file.lower() == 'cam':
        input_feeder = InputFeeder(input_type='cam')
    else:
        if not os.path.isfile(input_file):
            logger.error("Unable to find video file for input")
            exit(1)
        input_feeder = InputFeeder(input_type='video', input_file=input_file)

    input_feeder.load_data()
    width = int(input_feeder.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_feeder.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_feeder.cap.get(cv2.CAP_PROP_FPS))

    writer = None
    green_color = (0,255,0)
    blue_color = (255,0,0)
    red_color = (0, 0, 255)

    frame_counter = 0
    start_inference = time.time()

    for ret, frame in input_feeder.next_batch():
        if not ret:
            break

        frame_counter += 1

        image = cv2.resize(frame, (width, height))

        face_coordinates = face_detection_process(face_model, image)
        
        cv2.rectangle(image, (face_coordinates[0], face_coordinates[1]), (face_coordinates[2], face_coordinates[3]), green_color, 2)

        cropped_face_image = frame.copy()

        cropped_face_image = cropped_face_image[face_coordinates[1]:face_coordinates[3], face_coordinates[0]:face_coordinates[2]]

        head_poses = head_pose_process(head_pose_model, cropped_face_image)

        # print("[INFO][Head Pose] pitch: {}, roll: {}, yaw: {}".format(head_poses[0], head_poses[1], head_poses[2]))

        facial_landmark, left_eye_coord, right_eye_coord = facial_landmark_process(facial_landmark_model, cropped_face_image)

        for (x, y) in facial_landmark:
            cv2.circle(cropped_face_image, (x, y), 3, red_color, -1)
        
        cv2.rectangle(cropped_face_image, left_eye_coord[0], left_eye_coord[1], green_color, 2)
        cv2.rectangle(cropped_face_image, right_eye_coord[0], right_eye_coord[1], green_color, 2)

        cropped_left_eye_image = cropped_face_image.copy()
        cropped_left_eye_image = cropped_left_eye_image[left_eye_coord[0][1]:left_eye_coord[1][1], left_eye_coord[0][0]:left_eye_coord[1][0]]

        cropped_right_eye_image = cropped_face_image.copy()
        cropped_right_eye_image = cropped_right_eye_image[right_eye_coord[0][1]:right_eye_coord[1][1], right_eye_coord[0][0]:right_eye_coord[1][0]]

        gaze_angles = gaze_estimation_process(gaze_estimation_model, head_poses, cropped_left_eye_image, cropped_right_eye_image)

        # print("[INFO][Gaze Angles] X: {}, Y: {}, Z: {}".format(gaze_angles[0], gaze_angles[1], gaze_angles[2]))

        left_eye_center_coords = (int((left_eye_coord[0][0] + left_eye_coord[1][0])/2), int((left_eye_coord[0][1] + left_eye_coord[1][1])/2))
        new_left_eye_coords = (int(left_eye_center_coords[0] + gaze_angles[0]*90), int(left_eye_center_coords[1]+ gaze_angles[1]*90*-1))

        right_eye_center_coords = (int((right_eye_coord[0][0] + right_eye_coord[1][0])/2), int((right_eye_coord[0][1] + right_eye_coord[1][1])/2))
        new_right_eye_coords = (int(right_eye_center_coords[0] + gaze_angles[0]*90), int(right_eye_center_coords[1]+ gaze_angles[1]*90*-1))

        cv2.arrowedLine(cropped_face_image, left_eye_center_coords, new_left_eye_coords, blue_color, 3)
        cv2.arrowedLine(cropped_face_image, right_eye_center_coords, new_right_eye_coords, blue_color, 3)

        cv2.rectangle(image, (-10,50), (700, 50), (255,255,255), 120)
        cv2.putText(image, "pitch: {:.2f}, roll: {:.2f}, yaw: {:.2f}".format(head_poses[0], head_poses[1], head_poses[2]), (20,30), cv2.FONT_HERSHEY_DUPLEX, 1, red_color, 3)
        cv2.putText(image, "X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(gaze_angles[0], gaze_angles[1], gaze_angles[2]), (20,100), cv2.FONT_HERSHEY_DUPLEX, 1, blue_color, 3)

        mouse_controller.move(gaze_angles[0], gaze_angles[1])

        image = np.hstack((cv2.resize(image, (500, 500)), cv2.resize(cropped_face_image, (500, 500))))

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args.output, fourcc, 10, (image.shape[1], image.shape[0]), True)
        
        writer.write(image)

        cv2.imshow('face detection', image)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    total_inference_time = round(time.time() - start_inference, 1)
    fps = int(frame_counter) / total_inference_time

    print('[INFO] Face Detection model loading time = {} ms'.format(face_loading_time))
    print('[INFO] Head Pose Estimation model loading time = {} ms'.format(head_pose_time))
    print('[INFO] Facial Landmark detection model loading time = {} ms'.format(facial_landmark_time))
    print('[INFO] Gaze Estimation model loading time = {} ms'.format(gaze_model_time))
    print('[INFO] Total time to load all model = {} ms'.format(total_loading_time))

    print("[INFO] Total inference time = {} s".format(total_inference_time))
    print("[INFO] FPS =", fps)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark.txt'), 'w') as f:
        f.write('Face Detection model loading time:' + str(face_loading_time)+'ms'+'\n')
        f.write('Head Pose Estimation model loading time:' + str(head_pose_time)+'ms'+'\n')
        f.write('Facial Landmark detection model loading time:' + str(facial_landmark_time)+'ms'+'\n')
        f.write('Gaze Estimation model loading time:' + str(gaze_model_time)+'ms'+'\n')
        f.write('Total time to load all model:'+str(total_loading_time)+'ms'+'\n')
        f.write('Total inference time : ' + str(total_inference_time) + 's'+ '\n')
        f.write('fps: ' + str(fps) + '\n')
        
    input_feeder.close()
    cv2.destroyAllWindows()

    


