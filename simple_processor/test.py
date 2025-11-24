from raw_process import RawProcess
from pipeline import Pipeline


 # _2u0MkRqpjA, 0.3333333432674408,1.0,0.0,0.3333333432674408,0.0,0.0,0.0
data_path, data_name = "processed_data", "sample_video_5"
device = 'cuda'
video_path = "raw/5.mp4" # path to input video file

# /data1/9th-ABAW/output/resnet18_scheduler_fold0/best.pt / resnet18
# /data1/9th-ABAW/output/mobilenet_scheduler_fold0/best.pt / mobilenet
# /data1/9th-ABAW/output/lr1e-5_fold0/best.pt / resnet50
# /data1/9th-ABAW/output/rflf_scheduler_fold0/best.pt / resnet50 / rflf
# /data1/9th-ABAW/output/schedule_fold0/best.pt / resnet50
# /data1/9th-ABAW/output/scheduler_sg_fold0/best.pt / resnet50 / sg
# /data1/9th-ABAW/output/efficientnet_scheduler_fold0/best.pt /efficientnet
backbone_path = "/data1/9th-ABAW/output/efficientnet_sg_fold0/epoch_4.pt"
backbone_setting = {
    'simple_gate': True,
    'visual_backbone_type': 'efficientnet'
}


pipeline = Pipeline(device, data_path, data_name, backbone_path, backbone_setting)
result0, result_rest = pipeline(video_path)
levels0 = [
    -3.00, -2.66, -2.33, -2.00, -1.66, -1.33, -1.00, -0.66, -0.33,
    0.00,
    0.33,  0.66,  1.00,  1.33,  1.66,  2.00,  2.33,  2.66,  3.00
]  # 19개

levels_rest = [0.00, 0.33, 0.66, 1.00]  # 4개

emotion0 = levels0[result0.argmax(dim=-1).item()]
emotion_rest = [levels_rest[i] for i in result_rest.argmax(dim=-1).tolist()]

print("Predicted emotion (Arousal/Valence):", emotion0)
print("Predicted emotion (Expressions):", emotion_rest)

# print("Pipeline output:", result.shape) # emotion
# print(result)