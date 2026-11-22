# Plant Disease Detection — 39-Class CNN & Streamlit Demo

An educational plant-leaf image-classification project based on the original 39-class PlantVillage CNN implemented with PyTorch. This repository includes the original Flask interface and notebook, plus a maintained Streamlit demo with safer image handling and optional loading of a trained CNN checkpoint.

> **Important:** This is an educational image-classification demo, not an agronomic diagnostic service. Results can be wrong, especially for field photos, unfamiliar cultivars, mixed symptoms, different cameras or backgrounds. A model score is not a calibrated probability. Confirm suspected diseases with a qualified local agricultural expert before applying treatments or pesticides.

## Demo

- 🌱live demo: https://plant-disease-detection-ny4zezbmkqsrisxvckitqm.streamlit.app/
- **Repository:** https://github.com/Kamleshkumarmandal/Plant-Disease-Detection

### Deploy the Streamlit preview

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io) with GitHub.
2. Select **Create app** and choose this repository.
3. Repository: `Kamleshkumarmandal/Plant-Disease-Detection`.
4. Branch: `main`.
5. Main file path: `streamlit_app.py`.
6. Click **Deploy**.

The public repository does not contain the large trained `.pt` checkpoint. The Streamlit app therefore starts in **preview-only mode**: image uploads, image previews and the disease reference library work, but predictions remain disabled until a trusted model checkpoint is supplied. Streamlit Cloud will generate the public `https://....streamlit.app` address after deployment; add that generated URL here once it exists.

## What is included

### Streamlit demo (root of repository)

- Upload JPG, JPEG and PNG plant-leaf images (10 MB upload limit).
- Validate image files and preview them without saving uploads on the server.
- Display the model's top-ranked classes when a trained model is configured.
- Show dataset reference descriptions and notes for the predicted class.
- Browse the 39 PlantVillage class labels and disease descriptions.
- Clearly indicate when model weights are unavailable rather than showing a fabricated prediction.

### Original Flask application

The existing Flask application under `Plant-Disease-Detection-main/Flask Deployed App` is retained. Its inference path now loads the model only when required, maps checkpoint tensors to CPU, validates uploaded file type and size, processes images in memory, resolves data/model paths relative to the application folder and runs with debug mode disabled by default.

### Model and class mapping

- `plant_disease_model.py`: shared implementation of the original 39-class CNN architecture.
- `plant_disease_labels.py`: stable 39-index class mapping.
- `inference.py`: preprocessing, checkpoint validation and top-k prediction helpers.
- `disease_info.csv`: reference descriptions for the 39 model classes.
- `requirements-model.txt`: optional PyTorch dependency set for running model inference.

## Run the Streamlit preview locally (Windows PowerShell)

Use Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit. The preview app works without PyTorch or model weights.

## Enable CNN predictions locally

1. Download the original trained checkpoint from the model link referenced in the project:
   https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link
2. Verify that the file is a trusted state-dictionary checkpoint for this repository's 39-class CNN.
3. Install optional inference dependencies:

```powershell
pip install -r requirements-model.txt
```

4. Set the model path for the current PowerShell session:

```powershell
$env:PLANT_DISEASE_MODEL_PATH="C:\path\to\plant_disease_model_1_latest.pt"
streamlit run streamlit_app.py
```

The app uses CPU inference. The checkpoint is not included in Git and `.pt` files are excluded by `.gitignore`. Do not load model files from unknown sources: PyTorch checkpoint files are serialized objects and must be treated as trusted inputs.

For a hosted prediction demo, arrange a secure way to provide a trusted model artifact to the hosting service and install the optional PyTorch dependencies there. Do not put large model files, patient data or other sensitive files into a public repository.

## Run the original Flask app locally

The Flask app has its own dependency file. In PowerShell:

```powershell
pip install -r "Plant-Disease-Detection-main/Flask Deployed App/requirements.txt"
cd "Plant-Disease-Detection-main/Flask Deployed App"
$env:PLANT_DISEASE_MODEL_PATH="C:\path\to\plant_disease_model_1_latest.pt"
python app.py
```

Then open http://127.0.0.1:5000. If the checkpoint is missing, the app now displays a clear message instead of failing during startup.

## Dataset and original notebook

The original notebook and documentation remain under `Plant-Disease-Detection-main/Model`. The project references the PlantVillage dataset in its original materials:

- https://data.mendeley.com/datasets/tywbtsjrjv/1
- https://www.kaggle.com/datasets/emmarex/plantdisease

Check the original notebook's class-folder ordering and preprocessing assumptions when using a different dataset version. The included class labels are tied to the original 39-output CNN checkpoint.

## Tests

The lightweight Streamlit preview CI checks Python compilation, the 39-class mapping, the 39-row disease information table and a Streamlit render smoke test. It does not download the dataset, install PyTorch, train the CNN or validate predictive accuracy.

Run the local preview tests:

```powershell
pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
```

GitHub Actions runs those checks on pushes and pull requests.

## Repository structure

```text
streamlit_app.py             # Streamlit demo
plant_disease_model.py       # Original CNN architecture
plant_disease_labels.py      # 39-class model label mapping
inference.py                 # Safe checkpoint loading and preprocessing
disease_info.csv             # Disease reference descriptions
requirements.txt             # Lightweight Streamlit preview dependencies
requirements-model.txt       # Optional PyTorch inference dependencies
tests/                       # Metadata and Streamlit smoke tests
Plant-Disease-Detection-main/
  Flask Deployed App/        # Hardened legacy Flask application
  Model/                     # Original notebook, documentation and model notes
  test_images/               # Sample test images
  demo_images/               # Original screenshots
```

## Known limitations and future improvements

- The model checkpoint is an external file and must be supplied separately.
- The original CNN was trained on PlantVillage-style images; real field images may be substantially different.
- Softmax scores are shown only as model ranking scores; they are not calibrated confidence values.
- The CSV descriptions and treatment notes are reference material from the original project and may not be current or locally appropriate.
- This project has not been clinically or agronomically validated.
- Future work: reproducible training script, independent test-set report, field-image evaluation, calibration, Grad-CAM explainability, and reviewed region-specific agronomy guidance.

## Original project credits

The original model, Flask app, notebook, dataset references and interface materials are retained in the nested project directory. This repository's Streamlit and safety/reliability updates are additional maintenance work.
