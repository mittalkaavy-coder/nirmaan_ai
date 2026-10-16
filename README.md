# NIRMAAN AI

AI-powered waste intelligence platform for detecting and analyzing waste, prioritizing cleanup, and supporting hotspot monitoring.

## Core Workflow

Detect -> Analyze -> Prioritize -> Assign -> Clean -> Verify -> Predict

## Features

- Waste classification using a trained TensorFlow/Keras model.
- Waste detection using a trained YOLO model.
- FastAPI backend for serving waste-analysis endpoints.
- React and Vite frontend.
- API documentation through FastAPI Swagger UI.

## Project Structure

```text
nirmaan_ai/
├── ai/
│   ├── models/
│   └── training/
├── backend/
│   └── app/
├── frontend/
├── runs/
├── requirements.txt
└── README.md
```

## Prerequisites

- Python compatible with the installed backend dependencies.
- Node.js and npm.
- Git and Git LFS.

## Backend Setup (Windows PowerShell)

From the repository root, create and activate a virtual environment if you do not already have one:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the API:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open the API documentation:

http://127.0.0.1:8000/docs

Health endpoint:

http://127.0.0.1:8000/health

## Frontend Setup

Open a second terminal and run:

```powershell
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite, usually:

http://localhost:5173

Build the frontend for production:

```powershell
npm run build
```

## Model Files

The application uses trained model checkpoints. Large model files are tracked with Git LFS.

Keep the required model files at the paths expected by the backend. Do not commit datasets, virtual environments, `node_modules`, or generated build output.

## API Endpoints

- `GET /`
- `GET /health`
- `POST /api/v1/waste/analyze`
- `POST /api/v1/waste/analyze-image`
- `POST /api/v1/waste/detect-image`

Refer to `/docs` for the request schemas and available responses.

## Project Status

This repository contains the application source code, model checkpoints, and frontend needed to develop and test the NIRMAAN AI prototype.
