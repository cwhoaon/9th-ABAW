import cv2
import os
import sys
import subprocess
import numpy as np
from pathlib import Path
import wave, json
from vosk import Model, KaldiRecognizer
import soundfile as sf
import pandas as pd

from vggish import vggish_input
from facial_landmark import facial_image_crop_by_landmark, FacenetController
from vggish.vggish import VGGish
from vggish.hubconf import model_urls
from facenet_pytorch import MTCNN
from deepmultilingualpunctuation import PunctuationModel
from speech import extract_transcript

class RawProcess:
    def __init__(self, device, data_path, data_name):
        super().__init__()
        self.device = device
        # self.video_path = video_path
        self.data_path = data_path
        self.data_name = data_name

        # video = cv2.VideoCapture(self.video_path)

        # self.wav_path = os.path.join(self.data_path, self.data_name + ".wav")
        # self.vggish_path = os.path.join(self.data_path, "vggish", self.data_name + ".csv")

        # self.fps = video.get(cv2.CAP_PROP_FPS)
        # self.frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.vggish_model = self.load_vggish_model()
        self.mtcnn, self.face_landmark_detector = self.load_mtcnn_model()
        self.model = Model("/data1/9th-ABAW/preprocessing/load/vosk-model-en-us-0.22")

        

    def preprocess(self, video_path):
        self.video_path = video_path
        video = cv2.VideoCapture(self.video_path)

        self.wav_path = os.path.join(self.data_path, self.data_name + ".wav")
        self.vggish_path = os.path.join(self.data_path, "vggish", self.data_name + ".csv")

        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.wav_path = self.convert_video_to_wav(self.video_path, self.data_path, self.data_name, target_frequency=16000)
        # new로 전달, vgg csv는 저장 안할거
        self.vggish_path, new = self.extract_vggish(self.wav_path, self.data_path, self.data_name, self.vggish_model)
        # 이게 너무오래걸림;;
        success = self.crop_align_face_fn(self.video_path, self.data_path, self.data_name, mtcnn=self.mtcnn, face_landmark_detector=self.face_landmark_detector)
        if not success:
            exit(1)
        # 이거도 저장안하고 string으로 바로 전달
        # punctuation 하면 시간 엄청걸림
        extract_transcript(self.wav_path, self.data_path, self.data_name, self.model, KaldiRecognizer)
        
        print("Preprocessing completed.")
        return new
    
    def load_vggish_model(self):
        vggish_model = VGGish(device=self.device, urls=model_urls, pretrained=True, preprocess=False, postprocess=False, progress=True)
        vggish_model.eval()
        return vggish_model

    def load_mtcnn_model(self):
        import face_alignment
        face_landmark_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=self.device)
        mtcnn = MTCNN(keep_all=False, device=self.device, image_size=128, margin=12, select_largest=False, post_process=False)
        return mtcnn, face_landmark_detector
    
    def load_punctuation_model(self):
        punc_cap_model = PunctuationModel(model="oliverguhr/fullstop-punctuation-multilang-large")

    def convert_video_to_wav(self, input_path, data_path, data_name, target_frequency=16000):
        output_path = os.path.join(data_path, data_name, "audio.wav")

        # if os.path.isfile(output_path):
        #     print("WAV file already exists, skipping conversion.")
        #     return output_path
        
        self.ensure_dir(output_path)
        # ffmpeg command to execute
        # -ac 1 for mono, -ar 16000 for sample rate 16k, -q:v 0 for keeping the quality.
        ffmpeg_command = "ffmpeg -i {input_path} -ac 1 -ar {frequency}  -q:v 0 -f wav {output_path}".format(
        input_path=input_path, output_path=output_path, frequency=target_frequency)

        full_command = "export PATH={conda_path}/bin:$PATH && {ffmpeg_command}".format(conda_path=sys.exec_prefix,
                                                                                   ffmpeg_command=ffmpeg_command)
        # execute if the output does not exist
        if not os.path.isfile(output_path):
            subprocess.call(full_command, shell=True)
        
        return output_path
    
    def extract_vggish(self, input_path, data_path, data_name, vggish_model):
        hop_sec = 1 / self.fps

        output_path = os.path.join(data_path, data_name, "vggish.csv")

        # if os.path.isfile(output_path):
        #     print("VGGish features already exist, skipping extraction.")
        #     return output_path

        examples_batch = vggish_input.wavfile_to_examples(input_path, window_sec=0.96, hop_sec=hop_sec)

        examples_segment = []
        examples_output = []
        input_size = 100
        if len(examples_batch) > input_size:
            num_segments = len(examples_batch) // input_size
            for i in range(num_segments):
                start = i * input_size
                end = (i + 1) * input_size
                if i == num_segments - 1:
                    end = len(examples_batch)
                examples_segment.append(examples_batch[start:end])
        else:
            examples_segment = [examples_batch]

        for example in examples_segment:
            vggish_feature = vggish_model.forward(example)
            examples_output.append(vggish_feature.cpu().detach().numpy())

        examples_output = np.vstack(examples_output)
        np.savetxt(output_path, examples_output, delimiter=";")

        return output_path, examples_output

    def crop_align_face_fn(self, input_path, data_path, data_name, mtcnn=None, face_landmark_detector=None):
        fnet = FacenetController(
            mtcnn=mtcnn, face_landmark_detector=face_landmark_detector,
            device=self.device, image_size=128, batch_size=128, input_path=input_path,
            output_path=os.path.join(data_path, data_name),
        )

        dataloader = fnet.get_dataloader()
        
        file_name = "video"
        csv_path = os.path.join(data_path, data_name, file_name + ".csv")

        if os.path.isfile(csv_path) and os.path.isdir(os.path.join(data_path, data_name, "frames")):
            print("Cropped face already exists, skipping processing.")
            return True

        output_path = fnet.process(dataloader, csv_path)



        if output_path == -1:
            print("Facial landmark detection failed")
            return False
        
        return True
    
    def ensure_dir(self, file_path):
        directory = file_path
        if file_path[-3] == "." or file_path[-4] == ".":
            directory = os.path.dirname(file_path)
        Path(directory).mkdir(parents=True, exist_ok=True)
