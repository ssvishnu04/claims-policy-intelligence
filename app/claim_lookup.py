import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from app.config import RAW_DATA_DIR


def load_claims_history() -> pd.DataFrame:
    path = Path(RAW_DATA_DIR) / "claims_history" / "claims_history.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_repair_estimates() -> pd.DataFrame:
    path = Path(RAW_DATA_DIR) / "repair_estimates" / "repair_estimates.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_fnol_record(claim_id: str) -> Dict[str, Any]:
    fnol_dir = Path(RAW_DATA_DIR) / "fnol"

    if not fnol_dir.exists():
        return {}

    for file in fnol_dir.glob("*.json"):
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
            if str(payload.get("claim_id", "")).lower() == claim_id.lower():
                return payload
        except Exception:
            continue

    return {}


def get_claim_history_record(policy_id: str, claim_id: str) -> Dict[str, Any]:
    df = load_claims_history()

    if df.empty:
        return {}

    filtered = df[
        (df["policy_id"].astype(str).str.lower() == policy_id.lower())
        & (df["claim_id"].astype(str).str.lower() == claim_id.lower())
    ]

    if filtered.empty:
        return {}

    return filtered.iloc[0].to_dict()


def get_repair_estimates(claim_id: str) -> List[Dict[str, Any]]:
    df = load_repair_estimates()

    if df.empty:
        return []

    filtered = df[df["claim_id"].astype(str).str.lower() == claim_id.lower()]

    if filtered.empty:
        return []

    return filtered.to_dict(orient="records")


def get_claim_profile(policy_id: str, claim_id: str) -> Dict[str, Any]:
    fnol = load_fnol_record(claim_id)
    history = get_claim_history_record(policy_id, claim_id)
    estimates = get_repair_estimates(claim_id)

    normalized_history = {
        "claim_id": claim_id,
        "policy_id": policy_id,
        "loss_date": history.get("loss_date") or history.get("date") or fnol.get("loss_date") or "Not Available",
        "loss_type": history.get("loss_type") or fnol.get("loss_type") or "Not Available",
        "claim_status": history.get("claim_status") or history.get("status") or fnol.get("status") or "Not Available",
        "paid_amount": history.get("paid_amount") or "Not Available",
        "description": history.get("description") or fnol.get("description") or "Not Available",
        "state": history.get("state") or fnol.get("state") or "Not Available",
    }

    filtered_estimates = [
        e for e in estimates
        if str(e.get("claim_id", "")).lower() == claim_id.lower()
    ]

    estimate_total = 0
    for row in filtered_estimates:
        try:
            estimate_total += float(row.get("estimated_cost", 0))
        except Exception:
            pass

    return {
        "policy_id": policy_id,
        "claim_id": claim_id,
        "fnol": fnol,
        "claims_history": normalized_history,
        "repair_estimates": filtered_estimates,
        "repair_estimate_total": estimate_total,
    }


def format_claim_profile_for_prompt(profile: Dict[str, Any]) -> str:
    return f"""
STRUCTURED CLAIM PROFILE

Policy ID: {profile.get("policy_id")}
Claim ID: {profile.get("claim_id")}

Claims History:
{json.dumps(profile.get("claims_history", {}), indent=2)}

FNOL:
{json.dumps(profile.get("fnol", {}), indent=2)}

Repair Estimates:
{json.dumps(profile.get("repair_estimates", []), indent=2)}

Repair Estimate Total:
{profile.get("repair_estimate_total")}
"""