# Original Plant Disease Detection Project

The maintained project setup, Streamlit demo instructions, model configuration and safety notes are now documented in the repository-root [README](../README.md).

## Current app entrypoints

- **Streamlit demo:** `streamlit_app.py` at the repository root.
- **Legacy Flask application:** `Flask Deployed App/app.py` in this folder.
- **Original CNN model and notebook:** `Model/`.

The trained PyTorch checkpoint is not stored in this public repository. Streamlit runs in preview-only mode until a trusted checkpoint is supplied. See the root README for the exact `PLANT_DISEASE_MODEL_PATH` setup.

## Original references

- Dataset: https://data.mendeley.com/datasets/tywbtsjrjv/1
- Original model article: https://medium.com/analytics-vidhya/plant-disease-detection-using-convolutional-neural-networks-and-pytorch-87c00c54c88f

This is an educational classifier, not a validated agricultural diagnostic service. Confirm suspected plant diseases with a qualified local agricultural expert before applying treatments.
