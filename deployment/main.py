from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import joblib
import tempfile
import os

from features import featurize_for_model
from audio_convert import convert_to_wav

app = FastAPI(title="Music Genre Classifier API")

# Loaded once at startup, not per-request to save time and compute
model = None
scaler = None
label_encoder = None
feature_columns = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler, label_encoder, feature_columns
    model = joblib.load("artifacts/model.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
    label_encoder = joblib.load("artifacts/label_encoder.pkl")
    feature_columns = joblib.load("artifacts/feature_columns.pkl")
    print("Artifacts loaded successfully.")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Music Genre Classifier API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_extensions = {".mp4", ".wav", ".mp3", ".m4a"}
    suffix = os.path.splitext(file.filename)[1].lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_extensions}",
        )

    tmp_path = None
    wav_path = None

    try:
        # 1. Save the upload to a temp file so PyAV can open it from disk
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # 2. Convert to a clean mono 22050Hz wav (handles mp4, mp3, m4a, wav)
        wav_path = convert_to_wav(tmp_path)

        # 3. Extract features in the exact column order the model was trained on
        features_df = featurize_for_model(wav_path, feature_columns)

        # 4. Scale using the SAME fitted scaler from training
        features_scaled = scaler.transform(features_df)

        # 5. Predict
        prediction = model.predict(features_scaled)
        genre = label_encoder.inverse_transform(prediction)[0]

        response = {"predicted_genre": genre}

        # Optional: include per-class confidence if the model supports it
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_scaled)[0]
            response["confidence"] = {
                cls: round(float(p), 4)
                for cls, p in zip(label_encoder.classes_, probs)
            }

        return JSONResponse(content=response)

    except RuntimeError as e:
        # Raised by convert_to_wav on decode failures (corrupt file, no audio stream, etc.)
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    finally:
        # Clean up temp files regardless of success/failure
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)