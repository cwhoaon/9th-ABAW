# 9th-ABAW Competition

This repository contains the implementation of our models for Valence-Arousal estimation, submitted to the 9th Affective Behavior Analysis in the Wild (ABAW) Challenge.
We introduce a TAGF (Time-aware Gated Fusion) model that incorporates temporal dynamics into the fusion process to enhance multimodal robustness and stability.



## 📦 Code for Preprocessing

The code for preprocessing is provided in the `preprocessing` folder.



## ⚙️ Specify the Settings

In `main.py`:

- Adjust the four paths in **1.2** for your machine.
- In **1.3**, name your experiment by specifying the `-stamp` argument.  
  When running `main.py`, carefully name your instance.  
  The name determines the output directory. If two runs have the same name, the latter will overwrite the earlier one.
- In **1.4**, to resume an instance, add `-resume 1` to the command:  
  - `python main.py -resume 0` will start a fresh run.  
  - `python main.py -resume 1` will resume an existing instance from the latest checkpoint.
- In **1.5**, for efficient debugging, specify `-debug 1` so that only one trial will be loaded per fold.
- In **1.7**, specify `-emotion` to either `arousal` or `valence`.
- In **2.1**, specify `-folds_to_run` from 0 to 6.  
  - Example: `-folds_to_run 0` runs fold 0.  
  - `-folds_to_run 0 1 2` runs fold 0, 1, and 2 in a row.



## 🚀 Run the Code

Usually, with all the default settings in `main.py` properly set, just run:

```bash
python main.py -folds_to_run 0 -emotion "valence" -stamp "cv"
```

To run multiple folds on multiple machines:

```bash
python main.py -folds_to_run 0 1 2 -emotion "valence" -stamp "cv"
```

If training stops unexpectedly, resume the latest epoch with:

```bash
python main.py -folds_to_run 0 -emotion "valence" -stamp "cv" -resume 1
```


## 📁 Collect the Result

The results will be saved in your specified -save_path, and will include:

- training logs (CSV format)
- trained model state dict and checkpoint
- predictions on the unseen partition



## 🙏 Acknowledgements

The code for preprocessing and the coding framework for the proposed model is based on
- https://github.com/praveena2j/ABAW8/tree/main
- https://github.com/sucv/ABAW2/tree/main
- https://github.com/sucv/ABAW3
