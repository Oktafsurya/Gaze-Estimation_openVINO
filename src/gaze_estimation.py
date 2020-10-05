import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import os

class GazeEstimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):

        model_xml, model_bin = str(model_name), str(os.path.splitext(model_name)[0] + ".bin")
        self.core = IECore()
        self.gaze_model = self.core.read_network(model=model_xml, weights = model_bin)
        self.input_blob = next(iter(self.gaze_model.inputs))
        self.out_blob = next(iter(self.gaze_model.outputs))

    def load_model(self):

        self.exec_net = self.core.load_network(network=self.gaze_model, device_name="CPU")

        return self.exec_net

    def sync_inference(self, head_pose, left_eye_image, right_eye_image):
        return self.exec_net.infer({"head_pose_angles": head_pose,
                                    "left_eye_image": left_eye_image,
                                    "right_eye_image": right_eye_image
                                    })
        
    def async_inference(self, head_pose, left_eye_image, right_eye_image, request_id=0):
        # create async network
        async_net = self.exec_net.start_async(request_id, inputs={"head_pose_angles": head_pose,
                                                                    "left_eye_image": left_eye_image,
                                                                    "right_eye_image": right_eye_image
                                                                    })

        # perform async inference
        output_blob = next(iter(async_net.outputs))
        status = async_net.requests[request_id].wait(-1)
        if status == 0:
            result = async_net.requests[request_id].outputs[output_blob]
        return result

    def check_model(self):

        supported_layer_map = self.core.query_network(network=self.gaze_model, device_name="CPU")
        supported_layers = supported_layer_map.keys()
        unsupported_layer_exists = False
        network_layers = self.gaze_model.layers.keys()
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
            print("[INFO][Gaze Estimation Model] All layers are suported")

    def preprocess_input(self, image):
        
        p = self.gaze_model.inputs[self.input_blob].shape
        image = cv2.resize(image, (60, 60))
        image = image.transpose(2,0,1)
        image = image.reshape(*p, 60,60)
        return image

    def preprocess_output(self, outputs):

        gaze_info = outputs['gaze_vector'][0]

        return gaze_info
