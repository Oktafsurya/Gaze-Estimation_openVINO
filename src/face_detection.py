import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import os

class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):

        self.threshold = threshold
        model_xml, model_bin = str(model_name), str(os.path.splitext(model_name)[0] + ".bin")
        self.core = IECore()
        self.face_model = self.core.read_network(model=model_xml, weights = model_bin)
        self.input_blob = next(iter(self.face_model.inputs))
        self.out_blob = next(iter(self.face_model.outputs))

    def load_model(self):
        
        self.exec_net = self.core.load_network(network=self.face_model, device_name="CPU")

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
        supported_layer_map = self.core.query_network(network=self.face_model, device_name="CPU")
        supported_layers = supported_layer_map.keys()
        unsupported_layer_exists = False
        network_layers = self.face_model.layers.keys()
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
            print("[INFO][Face Detection Model] All layers are suported")

    def preprocess_input(self, image):

        n, c, h, w = self.face_model.inputs[self.input_blob].shape

        image = cv2.resize(image, (w, h))
        image = image.transpose(2,0,1)
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, image, outputs):

        width = image.shape[1]
        height = image.shape[0]

        # faces_coordinates = []
        
        for i in range(len(outputs[0][0])):
            box = outputs[0][0][i]
            conf = box[2]
            
            if conf >= self.threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                # faces_coordinates.append([xmin, ymin, xmax, ymax])

        return [xmin, ymin, xmax, ymax]
