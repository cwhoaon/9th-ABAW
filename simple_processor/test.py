from raw_process import RawProcess
from pipeline import Pipeline
 
data_path, data_name = "processed_data", "sample_video"
device, video_path = "cuda:0", "raw/4.mp4"

pipeline = Pipeline(device, data_path, data_name)
result = pipeline(video_path)

print("Pipeline output:", result.shape)
import ipdb; ipdb.set_trace()