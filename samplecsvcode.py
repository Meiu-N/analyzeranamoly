# ======================================
# AGENTFLOW AI - CSV RISK ANALYZER
# UI Upload Page + /analyze API
# PII + Numeric Anomaly + Groq LLM report
# ======================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import re
from io import StringIO
import uvicorn
import requests
import os
from typing import Dict, List, Tuple, Any

app = FastAPI(title="AgentFlow AI - Intelligent CSV Risk Analyzer")

# ------------------------------
# API KEY CONFIG
# ------------------------------
# Preferred: set GROQ_API_KEY in environment.
# Fallback: placeholder string (fill locally; do NOT commit to git).
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "PUT_YOUR_GROQ_KEY_HERE").strip()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192").strip()
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "20"))
ANOMALY_ZSCORE = float(os.getenv("ANOMALY_ZSCORE", "3.0"))


# ------------------------------
# UI: Simple HTML Upload Page
# ------------------------------
UPLOAD_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>AgentFlow AI - CSV Risk Analyzer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 30px; }
    .card { max-width: 760px; padding: 20px; border: 1px solid #ddd; border-radius: 12px; }
    h2 { margin-top: 0; }
    .row { margin: 12px 0; }
    button { padding: 10px 16px; cursor: pointer; }
    pre { white-space: pre-wrap; background: #f7f7f7; padding: 12px; border-radius: 8px; }
    .hint { color: #555; font-size: 13px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>AgentFlow AI - Intelligent CSV Risk Analyzer</h2>
    <p class="hint">Upload a CSV to analyze PII + numeric anomalies and get an AI risk report.</p>

    <div class="row">
      <input type="file" id="fileInput" accept=".csv" />
    </div>

    <div class="row">
      <button onclick="upload()">Analyze</button>
    </div>

    <div class="row">
      <h3>Result</h3>
      <pre id="result">No result yet.</pre>
    </div>

    <p class="hint">
      API endpoints: <code>POST /analyze</code> | Swagger: <code>/docs</code>
    </p>
  </div>

<script>
async function upload() {
  const fileInput = document.getElementById('fileInput');
  const result = document.getElementById('result');

  if (!fileInput.files || fileInput.files.length === 0) {
    result.textContent = "Please select a CSV file first.";
    return;
  }

  result.textContent = "Analyzing...";

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const resp = await fetch("/analyze", { method: "POST", body: formData });
    const data = await resp.json();
    result.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    result.textContent = "Error calling /analyze: " + e;
  }
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    # Browser-friendly upload UI
    return UPLOAD_PAGE


# ------------------------------
# Helpers
# ------------------------------
def _safe_read_csv(file_bytes: bytes) -> pd.DataFrame:
    """Robust CSV decode/load."""
    try:
        text = file_bytes.decode("utf-8", errors="replace")
        return pd.read_csv(StringIO(text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read CSV: {str(e)}")


# ------------------------------
# PII Detection (vectorized)
# ------------------------------
def detect_pii_columns(df: pd.DataFrame) -> List[str]:
    pii_columns: List[str] = []

    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"\b\d{10}\b"

    for col in df.columns:
        s = df[col].astype(str)
        if s.str.contains(email_pattern, regex=True, na=False).any():
            pii_columns.append(col)
            continue
        if s.str.contains(phone_pattern, regex=True, na=False).any():
            pii_columns.append(col)
            continue

    return sorted(list(set(pii_columns)))


# ------------------------------
# Numeric anomalies (3-sigma)
# ------------------------------
def detect_column_anomalies(df: pd.DataFrame, zscore_threshold: float = 3.0) -> Tuple[Dict[str, List[int]], List[int]]:
    anomaly_report: Dict[str, List[int]] = {}
    total_anomalies = set()

    numeric_df = df.select_dtypes(include=["number"])

    for col in numeric_df.columns:
        col_data = pd.to_numeric(numeric_df[col], errors="coerce")
        mean = col_data.mean()
        std = col_data.std()

        if std == 0 or pd.isna(std):
            continue

        mask = (col_data > mean + zscore_threshold * std) | (col_data < mean - zscore_threshold * std)
        idxs = df.index[mask.fillna(False)].tolist()

        if idxs:
            anomaly_report[col] = idxs
            total_anomalies.update(idxs)

    return anomaly_report, sorted(list(total_anomalies))


# ------------------------------
# Risk score
# ------------------------------
def calculate_risk_score(pii_count: int, anomaly_count: int) -> int:
    return pii_count * 3 + anomaly_count * 2


# ------------------------------
# Groq LLM report
# ------------------------------
def call_llm_agent(summary_text: str) -> str:
    if not GROQ_API_KEY or GROQ_API_KEY == "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":
        return "GROQ_API_KEY not configured. Set GROQ_API_KEY env var or replace placeholder locally."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = """
You are a Senior Data Risk Analyst AI.

Generate:
1. Executive Summary
2. Risk Severity (Low / Medium / High / Critical)
3. PII Risk Explanation
4. Anomaly Explanation
5. Business Impact
6. Recommended Remediation Steps

Rules:
- If no anomalies → dataset is stable.
- If anomaly rows > 5% → High risk.
- If PII exists + anomalies exist → Critical risk.
- Respond in clean markdown.
""".strip()

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary_text},
        ],
        "temperature": 0.1,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        return f"LLM Error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"LLM Connection Error: {str(e)}"


# ------------------------------
# API: Upload + Analyze
# ------------------------------
@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)) -> JSONResponse:
    content = await file.read()

    max_bytes = MAX_FILE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max allowed is {MAX_FILE_MB} MB.")

    df = _safe_read_csv(content)

    pii_columns = detect_pii_columns(df)
    column_anomalies, anomaly_rows = detect_column_anomalies(df, zscore_threshold=ANOMALY_ZSCORE)
    risk_score = calculate_risk_score(len(pii_columns), len(anomaly_rows))

    summary_text = f"""
Dataset Summary:
Total Rows: {len(df)}
Total Columns: {len(df.columns)}
Columns: {df.columns.tolist()}

PII Columns Detected: {pii_columns}

Total Anomaly Rows: {len(anomaly_rows)}
Column-wise Anomalies: {column_anomalies}

Risk Score: {risk_score}
""".strip()

    ai_report = call_llm_agent(summary_text)

    return JSONResponse(
        {
            "filename": file.filename,
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "columns": df.columns.tolist(),
            "pii_columns_detected": pii_columns,
            "column_anomalies": column_anomalies,
            "anomaly_rows": anomaly_rows,
            "total_anomalies": int(len(anomaly_rows)),
            "risk_score": int(risk_score),
            "ai_risk_report": ai_report,
        }
    )


# ------------------------------
# Run server
# ------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
