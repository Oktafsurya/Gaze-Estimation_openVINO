import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import os

class FacialLandmarkDetection:
    '''
    Class for the Facial Landmark Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):

        model_xml, model_bin = str(model_name), str(os.path.splitext(model_name)[0] + ".bin")
        self.core = IECore()
        self.facial_landmark_model = self.core.read_network(model=model_xml, weights = model_bin)
        self.input_blob = next(iter(self.facial_landmark_model.inputs))
        self.out_blob = next(iter(self.facial_landmark_model.outputs))

    def load_model(self):

        self.exec_net = self.core.load_network(network=self.facial_landmark_model, device_name="CPU")

        return self.exec_net

    def sync_inference(self, image):
        input_blob = next(iter(self.exec_net.inputs))
        return self.exec_net.infer({input_blob: image})
        
    def async_inference(self, image, request_id=0):
        # create async network
        input_blob = next(iter(self.exec_net.inputs))
        async_net = self.exec_net.start_async(request_id, inputs={input_blob: image})

        # perform async inference
        output_blob = next(iter(async_net.outputs))
        status = async_net.requests[request_id].wait(-1)
        if status == 0:
            result = async_net.requests[request_id].outputs[output_blob]
        return result

    def check_model(self):

        supported_layer_map = self.core.query_network(network=self.facial_landmark_model, device_name="CPU")
        supported_layers = supported_layer_map.keys()
        unsupported_layer_exists = False
        network_layers = self.facial_landmark_model.layers.keys()
        for layer in network_layers:
            if layer in supported_layers:
                pass
            else:
                print("[INFO] {} Still Unsupported".format(layer))
                unsupported_layer_exists = True
            
        if unsupported_layer_exists:
            print("Exiting the program.")
            exit(1)
        else: 
            print("[INFO][Facial Landmark Detection Model] All layers are suported")

    def preprocess_input(self, image):

        n, c, h, w = self.facial_landmark_model.inputs[self.input_blob].shape

        image = cv2.resize(image, (w, h))
        image = image.transpose(2,0,1)
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, image, outputs):

        width = image.shape[1]
        height = image.shape[0]

        # shape (1x70) 
        facial_landmark = outputs['align_fc3'][0] 
        
        # convert from [x0,y0,x1,y1,...,x34,y34] to [(x0,y0),(x1,y1),...,(x34,y34)] and scale to input size
        j = 0
        landmark_points = []
        for i in range(int(len(facial_landmark)/2)):
            point = (int(facial_landmark[j]*width), int(facial_landmark[j+1]*height))
            landmark_points.append(point)
            j += 2
        
        left_eye_coord = [(landmark_points[12][0], landmark_points[13][1]), (landmark_points[14][0], landmark_points[0][1]+30)]
        right_eye_coord = [(landmark_points[15][0], landmark_points[16][1]), (landmark_points[17][0], landmark_points[2][1]+30)]

        return landmark_points, left_eye_coord, right_eye_coord
