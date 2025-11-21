from raw_process import RawProcess
from pipeline import Pipeline
 
data_path, data_name = "processed_data", "sample_video"
device = 'cuda'
video_path = "raw/4.mp4" # path to input video file

pipeline = Pipeline(device, data_path, data_name)
result = pipeline(video_path)

print("Pipeline output:", result.shape) # emotion
import ipdb; ipdb.set_trace()