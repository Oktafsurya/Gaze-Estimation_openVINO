import os
import subprocess
import argparse
from pathlib import Path

def arg_parse():
   
    parser = argparse.ArgumentParser(description='Model downloader')

    parser.add_argument("--fd", dest = 'face_detection', help="face detection model", default="face-detection-adas-binary-0001")
    parser.add_argument("--fl", dest = 'facial_landmark', help="facial landmark model", default="facial-landmarks-35-adas-0002")
    parser.add_argument("--ge", dest = "gaze_estimation", help = "gaze estimation model", default = "gaze-estimation-adas-0002")
    parser.add_argument("--hp", dest = "head_pos", help = "head pose model", default = "head-pose-estimation-adas-0001")
    
    return parser.parse_args()

if __name__ ==  '__main__':

    args = arg_parse()

    output_folder="./models"
    assert "INTEL_OPENVINO_DIR" in os.environ, "[Error] OpenVINO environment not initialized"
    OPENVINO_dir = Path(os.environ["INTEL_OPENVINO_DIR"])
    downloader_script = OPENVINO_dir.joinpath("deployment_tools/tools/model_downloader/downloader.py")
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    models_used = [args.face_detection, args.facial_landmark, args.gaze_estimation, args.head_pos]

    for model in models_used:
        try:
            cmd = '''python3 "{}" --name {} -o {}'''.format(downloader_script, model.strip(), output_folder)
            cmd = " ".join([line.strip() for line in cmd.splitlines()])
            print(subprocess.check_output(cmd, shell=True).decode())
        except Exception as ex:
            print("Error downloading the model {} : {}".format(model, ex))
