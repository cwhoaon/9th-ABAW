import os
import shutil
from pathlib import Path
from base.utils import get_filename_from_a_folder_given_extension, get_all_files_recursively_by_ext
import numpy as np
import csv
import pandas as pd
import sys

arousal_result_path = "/home/dxlab/jupyter/Yubeen/save2/abaw9_LFAN_cv_fold0_arousal_seed3407/predict/extra/arousal"
valence_result_path = "/home/dxlab/jupyter/Yubeen/save4/abaw9_LFAN_cv_fold0_valence_seed3407/predict/extra/valence"


final_result_path = "TAGF_AV_results_bestaro"
os.makedirs(final_result_path, exist_ok= True)
test_set_list_path = "Valence_Arousal_Estimation_Challenge_test_set_release.txt"
test_set_list = pd.read_csv(test_set_list_path, header=None).values[:, 0]


sample_path = "ICCV_9th_ABAW_VA_test_set_sample.txt"
sample_df = pd.read_csv(sample_path)


for fold in range(6):
    fold =0
    valences, arousals, images = [], [], []
    for trial in test_set_list:
        arousal_txt = get_all_files_recursively_by_ext(arousal_result_path, "txt", trial)[0]
        valence_txt = get_all_files_recursively_by_ext(valence_result_path, "txt", trial)[0]
       
        assert "arousal" in arousal_txt and "valence" in valence_txt

        arousal = pd.read_csv(arousal_txt).values
        valence = pd.read_csv(valence_txt).values


        assert len(arousal) == len(valence)
        length = len(arousal)
        print(length)

        sample_length = len(sample_df[sample_df['image_location'].str.match(trial + "/")])

        diff = sample_length - length


        if diff > 0:

            length = sample_length
            arousal = np.concatenate((arousal, np.repeat(arousal[-1], diff)[:, np.newaxis]))
            valence = np.concatenate((valence, np.repeat(valence[-1], diff)[:, np.newaxis]))
        elif diff < 0:
            length = sample_length
            arousal = arousal[:sample_length, :]
            valence = valence[:sample_length, :]

        print("{} has diff = {} to {}".format(trial, str(diff), str(sample_length)))

        image = []
        for i in range(length):
            image.append(trial + "/" + str(i+1).zfill(5) + ".jpg")

        valences.extend(valence)
        arousals.extend(arousal)
        images.extend(image)

    result_path = os.path.join(final_result_path, "fold" + str(fold) + ".txt")
    result = np.c_[images, valences, arousals]

    result_df = pd.DataFrame(result, columns=["image_location","valence","arousal"])
    result_df.to_csv(result_path,sep=",", index=None)
    print("fold {} done!".format(str(fold)))
    sys.exit()
