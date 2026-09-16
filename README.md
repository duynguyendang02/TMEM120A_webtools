# TMEM120A Molecular Screening Tool

Streamlit web application for TMEM120A candidate prioritization using the frozen open-source PLS6 QSAR model.

## Model pipeline

SMILES -> RDKit + Mordred 2D descriptors -> frozen 931-feature preprocessing -> PLS6 -> predicted docking score/category.

The app also reports physicochemical properties, QED, medicinal-chemistry filters, and the nearest training analogue. An empirical descriptor-space OOD guard suppresses the docking score/category for extreme extrapolations.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

No Google Drive path is required. The required frozen model assets are bundled in `./model/`.

## Streamlit Community Cloud

1. Push the contents of this folder to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py`.
4. Deploy.

No secrets or environment variables are required for the bundled model.

## Scientific scope

The model was trained on 607 compounds and evaluated on an independent 202-compound test set. Final test metrics recorded for the frozen model are R2 = 0.5592, RMSE = 0.6328 kcal/mol, MAE = 0.5055 kcal/mol, and Spearman = 0.6643.

Predictions are intended for virtual screening and candidate prioritization, not as experimental binding affinities or replacements for molecular docking and experimental validation.
