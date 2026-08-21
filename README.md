# Music_Genre_Classification

This project implements a music genre classification system using handcrafted audio features (MFCCs, chroma, spectral, and rhythm descriptors) extracted from raw .wav files with librosa.

Random Forest and XGBoost classifiers were trained and evaluated for multi-class genre prediction, with XGBoost achieving a marginal (~0.5%) accuracy improvement over Random Forest. 

The trained XGBoost model is deployed as a FastAPI backend (with PyAV handling mp4/mp3/m4a → wav conversion) paired with a static frontend, hosted on Render and GitHub Pages respectively — try it live [here](https://plan28-06.github.io/Music_Genre_Classification/).
