"""FastAPI interface for molecular absorption wavelength prediction."""

import math
import sys

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rdkit import Chem
from uvicorn import run as app_run

from src.constants import (
    APP_HOST,
    APP_PORT,
    MOLECULE_SMILES_COLUMN,
    SOLVENT_SMILES_COLUMN,
)
from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.logger import logging
from src.pipline.training_pipeline import TrainPipeline

app = FastAPI(title="Molecular Absorption Wavelength Predictor")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def render_page(request: Request, **values):
    """Render the molecular prediction form and optional result/error."""
    return templates.TemplateResponse(
        request=request,
        name="spectroscopy.html",
        context={"chromophore": "", "solvent": "", **values},
    )


def validate_smiles(value: str, label: str) -> str:
    """Reject blank or chemically invalid SMILES before inference."""
    smiles = value.strip()
    if not smiles or Chem.MolFromSmiles(smiles) is None:
        raise ValueError(f"Please enter a valid {label} SMILES.")
    return smiles


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render_page(request)


@app.post("/", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    chromophore = str(form.get(MOLECULE_SMILES_COLUMN, ""))
    solvent = str(form.get(SOLVENT_SMILES_COLUMN, ""))
    values = {"chromophore": chromophore, "solvent": solvent}

    try:
        chromophore = validate_smiles(chromophore, "chromophore")
        solvent = validate_smiles(solvent, "solvent")
        inputs = pd.DataFrame([{
            MOLECULE_SMILES_COLUMN: chromophore,
            SOLVENT_SMILES_COLUMN: solvent,
        }])

        config = VehiclePredictorConfig()
        predictor = Proj1Estimator(
            bucket_name=config.model_bucket_name,
            model_path=config.model_file_path,
        )
        prediction_nm = float(predictor.predict(inputs)[0])
        if not math.isfinite(prediction_nm):
            raise ValueError("The model returned a non-finite prediction.")

        return render_page(request, prediction_nm=prediction_nm, **values)
    except ValueError as exc:
        return render_page(request, error=str(exc), **values)
    except Exception:
        logging.exception("Absorption wavelength prediction failed")
        return render_page(
            request,
            error="Prediction failed. Check that a trained model is deployed to S3.",
            **values,
        )


@app.get("/train", response_class=PlainTextResponse)
def train_model():
    """Start the configured training, evaluation, and deployment pipeline."""
    try:
        TrainPipeline().run_pipeline()
        return "Training pipeline finished. Check logs for model acceptance and deployment status."
    except Exception:
        logging.exception("Training pipeline failed")
        return PlainTextResponse("Training pipeline failed. Check server logs.", status_code=500)


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
