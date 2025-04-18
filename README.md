
Download python 3.10.9 in this location (C:\Users\DELL\AppData\Local\Programs\Python\Python313\)

The Structure of Folder should be like this
C:\Users\DELL\
└── sign-language-detector\
    ├── venv\                         ← Your virtual environment
    ├── train_model.py               ← Script to train your model
    ├── webcam_detection.py          ← Script to run webcam and detect signs
    ├── model.h5                     ← Trained model saved after training
    ├── SignLanguageDataset\         ← Dataset folder
    │   └── mini_asl_alphabet\       ← Actual dataset you downloaded
    │       └── asl_alphabet_train\  ← Contains folders A-Z with images


Download mini alphabet set from kiggle and extract that zip file LINK: https://www.kaggle.com/datasets/lcastrillon/mini-asl-alphabet

commands step by step (enter these commands into CMD) 
mkdir sign-language-detector
cd sign-language-detector
python -m venv venv
venv\Scripts\activate -------For Activate the virtual environment
pip install opencv-python tensorflow mediapipe numpy matplotlib streamlit
pip install flask
python train_model.py
pip install opencv-python opencv-python-headless
python webcam_detection.py
pip install tensorflow
pip install pyttsx3
pip install tensorflow opencv-python pyttsx3
pip install tensorflow opencv-python opencv-python-headless numpy
pip install Pillow


Press q to quit OR  ctrl + C 
