Face Recognition Lab — 50-person closed-set project
A local, consent-based machine-learning project that detects a face in an uploaded image, compares it to a private enrolled dataset, and returns a likely enrolled label or Unknown / low confidence.
How the ML works
Training a deep face encoder from scratch with only ~50 people is not a good first approach: it normally needs far more labeled images and compute. This project uses a pretrained face detector/embedding model, extracts a numerical vector for each face, and trains your own linear SVM identity classifier on the people in your dataset. The training script evaluates a hold-out test split and writes a metrics report.

This is a learning/research starter, not a production biometric system. A model trained on 50 people will only classify identities enrolled in this dataset; it cannot reliably identify arbitrary people on the internet. Unknown-person rejection is still imperfect, so never treat a result as proof of identity.
1. Requirements (Mac M2)
Use Python 3.11 or 3.12 in a virtual environment. Python 3.14 may not yet be supported by every computer-vision dependency.
cd face_recognition_lab
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
The first time the detector is initialized, InsightFace may download its model files, so an internet connection is needed for the first run.
Model licensing note
The InsightFace Python library code is MIT-licensed, but its bundled pretrained model files have separate terms and are listed as non-commercial research by default. This starter is for learning/research. Before any commercial or public deployment, review the model licensing and obtain the necessary rights or substitute a properly licensed model. See the links below.
2. Prepare the dataset
Add only images from people who have explicitly consented to be included. Use one folder per identity. Use stable labels such as person_001 if you do not need to expose names.
data/
├── faces/
│   ├── person_001/
│   │   ├── photo_001.jpg
│   │   ├── photo_002.jpg
│   │   └── ...
│   ├── person_002/
│   │   └── ...
│   └── ... (about 50 folders)
└── unknown/
    └── ... (optional calibration images of consenting people who are not enrolled)
Recommended starting target:
- 20–30+ images per person (50 people means roughly 1,000–1,500 images).
- Vary lighting, expression, distance, background, and head angle.
- Prefer images captured in multiple sessions. Avoid near-duplicate frames across train/test splits.
- Keep the face reasonably visible and avoid heavy filters, extreme blur, or tiny faces.
- For calibration, data/unknown/ can contain faces of consenting volunteers who are not enrolled under any identity folder. Those images are used only to help tune rejection of unknown people.
Please do not scrape social media, collect faces secretly, or use photos without permission. Keep the dataset and model files out of public Git repositories; the included .gitignore excludes them by default.
3. Train the classifier
From the project root, with the virtual environment active:
python train.py
The first run may download the model weights. Training extracts face embeddings, creates an 80/20 stratified split, trains a linear SVM, prints classification metrics, calibrates a similarity rejection threshold, and saves:
- models/face_classifier.joblib
- reports/training_metrics.json
- reports/confusion_matrix.csv
The script requires at least two identities and enough successfully detected face images per identity. Aim for 20–30+ images per person, even though the code's minimum is lower.
Evaluate honestly
Review the printed accuracy, per-class precision/recall, and confusion matrix. Accuracy on the hold-out split does not guarantee performance on new cameras, lighting, age ranges, or demographic groups. Test on completely separate capture sessions and on unknown volunteers. If accuracy is poor, improve the data before changing the model.
4. Start the website
streamlit run app.py
Streamlit will print a local URL, usually http://localhost:8501. Open it in your browser, upload an image, and choose Analyze face. The website chooses the largest face if the image contains multiple people.

For another training run, stop and restart the app if the cached model does not refresh.
Project structure
face_recognition_lab/
├── app.py                  # Upload website
├── train.py                # Dataset extraction, training and evaluation
├── requirements.txt
├── src/face_engine.py       # Detector, embeddings and prediction/rejection logic
├── data/faces/              # Enrolled identities (you supply these images)
├── data/unknown/            # Optional unknown-person calibration set
├── models/                  # Generated model bundle
└── reports/                 # Training evaluation outputs
Privacy and safe use
- Enroll only people who have agreed to face-based experimentation.
- Keep the site private; do not expose it as a public arbitrary-person identification service.
- Use neutral labels (person_001) unless displaying names is actually necessary.
- Provide a deletion process for a participant's images and retrain after removing them.
- Do not use this prototype for access control or any consequential decision.
- Images uploaded to the Streamlit app are processed in memory and are not intentionally written to disk by the app code; deployment infrastructure may have its own logging and retention behavior.
References
- InsightFace Python package and license notes
- InsightFace model zoo and model licensing
- InsightFace licensing information
