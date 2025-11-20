import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms
import torch
import csv
from tqdm import tqdm
from pathlib import Path

def facial_landmark_template():

    template = np.float32([
        (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
        (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
        (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
        (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
        (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
        (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
        (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
        (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
        (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
        (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
        (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
        (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
        (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
        (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
        (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
        (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
        (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
        (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
        (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
        (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
        (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
        (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
        (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
        (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
        (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
        (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
        (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
        (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
        (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
        (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
        (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
        (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
        (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
        (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

    template_min, template_max = np.min(template, axis=0), np.max(template, axis=0)
    template = (template - template_min) / (template_max - template_min)

    # Indices of inner eyes and bottom lip.
    key_indices = [39, 42, 57]

    # Indices of the outline.
    outline_indices = [*range(17), *range(26, 16, -1)]
    return template, key_indices, outline_indices


class facial_image_crop_by_landmark(object):
    def __init__(self, landmark_number=68, video_size=128, margin_size=0):
        self.template, self.template_key_indices, self.template_outline_indices = facial_landmark_template()
        self.landmark_number = landmark_number
        self.output_image_size = video_size

    def crop_image(self, image, landmark):
        affine_matrix = self.get_affine_matrix(landmark)
        aligned_image = self.align_image(image, affine_matrix)
        return aligned_image

    def align_image(self, image, affine_matrix):
        r'''
        Warp the frame by the defined affine transformation.
        :param frame: (uint8 ndarray), the frame to warp.
        :param affine_matrix: (float ndarray), the affine matrix.
        :return: (uint8 ndarray), the aligned frame.
        '''
        aligned_image = cv2.warpAffine(image, affine_matrix,
                                       (self.output_image_size,
                                        self.output_image_size))
        return aligned_image



    def get_affine_matrix(self, landmark):
        r"""
        Calculate the affine matrix from the source to the target coordinates.
            Here, the template_key_indices defines which points to select.
        :param landmark: (float ndarray), the landmark to align.
        :return: (float ndarray), the 2x3 affine matrix.
        """
        source = np.asarray(landmark[self.template_key_indices], dtype=np.float32)
        target = np.asarray(self.template[self.template_key_indices] * self.output_image_size, dtype=np.float32)
        affine_matrix = cv2.getAffineTransform(source, target)
        return affine_matrix
    

class FacenetController(object):
    def __init__(self, mtcnn, face_landmark_detector, device, image_size, batch_size, input_path, output_path):
        self.mtcnn = mtcnn
        self.face_landmark_detector = face_landmark_detector
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.input_path = input_path
        self.output_path = output_path


    def get_dataloader(self):
        import mmcv
        video = mmcv.VideoReader(self.input_path)
        frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
        indices = [i for i in range(video.frame_cnt)]
        sizes = [frame.size for frame in frames]

        dataloader = [[frames[i:i + self.batch_size], indices[i:i + self.batch_size], sizes[i:i + self.batch_size]] for i in range(0, len(frames), self.batch_size)]

        return dataloader

    def process(self, dataloader, csv_path):
        black_face = torch.zeros((3, self.image_size, self.image_size))
        csv_column = [
            'frame', 'face_id', 'time_stamp', 'confidence', 'success', *['x_' + str(i) for i in range(68)], *['y_' + str(i) for i in range(68)]
        ]

        success = True

        with open(csv_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(csv_column)

            for frames, idxes, sizes in tqdm(dataloader, total=len(dataloader)):
                try:
                    faces, probs = self.mtcnn(frames, return_prob=True)
                except Exception as e:
                    return -1

                for face, prob, idx, in zip(faces, probs, idxes):

                    success = 1

                    if face is None:
                        success = 0
                        prob = 0.0
                    else:
                        prob = prob[0]
                    
                    image_path = os.path.join(self.output_path, "frames", str(idx).zfill(5) + ".jpg")
                    ensure_dir(image_path)
                    if success:
                        face_PIL = transforms.ToPILImage()(face.to(torch.uint8))
                        face = face.permute(1, 2, 0).to(torch.uint8)
                        landmarks_list = self.face_landmark_detector.get_landmarks(face)
                        if landmarks_list is None:
                            landmarks_list = [np.zeros((68, 2), dtype=np.float32)]
                    else:
                        face_PIL = transforms.ToPILImage()(black_face)
                        landmarks_list = [np.zeros((68, 2), dtype=np.float32)]
                    face_PIL.save(image_path)

                    csv_row = [
                        str(idx), '0', '0', str(np.round(prob, 2)), str(success), *landmarks_list[0][:, 0], *landmarks_list[0][:, 1]
                    ]
                    
                    writer.writerow(csv_row)

        return self.output_path

def ensure_dir(file_path):
    directory = file_path
    if file_path[-3] == "." or file_path[-4] == ".":
        directory = os.path.dirname(file_path)
    Path(directory).mkdir(parents=True, exist_ok=True)