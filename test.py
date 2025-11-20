from simple_processor.raw_process import RawProcess
 
data_path, data_name = "/data1/9th-ABAW/simple_processor/data", "sample_video"
device, video_path = "cuda:0", "/data1/9th-ABAW/simple_processor/4.mp4"
x = RawProcess(device, video_path, data_path, data_name)
x.preprocess()