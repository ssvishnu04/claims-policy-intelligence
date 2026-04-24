from pathlib import Path
import json
import pandas as pd

BASE = Path("data/raw")

folders = [
    "policies",
    "fnol",
    "api_data",
    "claims_history",
    "adjuster_notes",
    "underwriting_guidelines",
    "repair_estimates",
    "supporting_documents",
]

for folder in folders:
    (BASE / folder).mkdir(parents=True, exist_ok=True)


# -----------------------------
# POLICIES - unstructured text
# -----------------------------
policies = {
    "homeowners_policy_pol_1001.txt": """
Policy ID: POL-1001
Product: Homeowners HO-3
State: Texas

COVERAGE TYPE: Water Damage
DEDUCTIBLE: $1,000
COVERAGE LIMIT: $50,000

WATER DAMAGE COVERAGE
We cover sudden and accidental discharge or overflow of water from within a plumbing system, heating system, air conditioning system, appliance, or fire sprinkler system.

EXCLUSIONS
We do not cover flood, surface water, storm surge, water below the surface of the ground, or repeated seepage or leakage over 14 days or more.

LOSS SETTLEMENT
Covered losses are settled based on actual cash value unless replacement cost coverage applies.
""",
    "personal_auto_policy_pol_2001.txt": """
Policy ID: POL-2001
Product: Personal Auto Policy
State: Texas

COVERAGE TYPE: Collision and Comprehensive Auto Coverage
COLLISION DEDUCTIBLE: $500
COMPREHENSIVE DEDUCTIBLE: $250
COVERAGE LIMIT: Actual Cash Value of Covered Auto

COLLISION COVERAGE
We cover direct and accidental loss to your covered auto caused by collision.

COMPREHENSIVE COVERAGE
We cover theft, fire, vandalism, hail, falling objects, animal impact, and glass breakage.

EXCLUSIONS
We do not cover mechanical breakdown, wear and tear, intentional damage, racing, or use of the vehicle for illegal activity.
""",
    "commercial_property_policy_pol_3001.txt": """
Policy ID: POL-3001
Product: Commercial Property Policy
State: Texas

COVERAGE TYPE: Commercial Property Damage
DEDUCTIBLE: $5,000
COVERAGE LIMIT: $500,000

PROPERTY COVERAGE
We cover direct physical loss to covered commercial property caused by covered causes of loss.

WIND AND HAIL
Wind and hail damage may be covered subject to deductible, roof condition, and policy terms.

EXCLUSIONS
We do not cover flood, earth movement, wear and tear, deterioration, or poor maintenance.
""",
    "business_owners_policy_pol_4001.txt": """
Policy ID: POL-4001
Product: Business Owners Policy
State: Texas

COVERAGE TYPE: Business Property and Liability
PROPERTY DEDUCTIBLE: $2,500
LIABILITY LIMIT: $1,000,000

BUSINESS PROPERTY
We cover direct physical loss to business personal property caused by a covered loss.

BUSINESS INTERRUPTION
We may cover loss of business income when operations are suspended due to a covered direct physical loss.

EXCLUSIONS
We do not cover flood, virus-related shutdowns, intentional acts, or normal wear and tear.
""",
}

for filename, content in policies.items():
    (BASE / "policies" / filename).write_text(content.strip(), encoding="utf-8")


# -----------------------------
# FNOL JSON - API-like events
# -----------------------------
claims = [
    {
        "claim_id": "CLM-2001",
        "policy_id": "POL-1001",
        "customer_id": "CUST-5001",
        "loss_date": "2024-03-10",
        "reported_date": "2024-03-11",
        "loss_type": "water_damage",
        "claim_status": "open",
        "paid_amount": 0,
        "description": "Sudden pipe burst in upstairs bathroom caused water damage to kitchen ceiling and flooring.",
        "state": "TX",
    },
    {
        "claim_id": "CLM-2002",
        "policy_id": "POL-1001",
        "customer_id": "CUST-5001",
        "loss_date": "2024-04-02",
        "reported_date": "2024-04-03",
        "loss_type": "flood_damage",
        "claim_status": "closed",
        "paid_amount": 0,
        "description": "Heavy rain caused surface water to enter the home through the front door.",
        "state": "TX",
    },
    {
        "claim_id": "CLM-2003",
        "policy_id": "POL-1001",
        "customer_id": "CUST-5002",
        "loss_date": "2024-06-18",
        "reported_date": "2024-06-19",
        "loss_type": "roof_wind_damage",
        "claim_status": "closed",
        "paid_amount": 4200,
        "description": "Windstorm damaged roof shingles on residential property.",
        "state": "TX",
    },
    {
        "claim_id": "CLM-3001",
        "policy_id": "POL-2001",
        "customer_id": "CUST-6001",
        "loss_date": "2024-01-22",
        "reported_date": "2024-01-22",
        "loss_type": "auto_collision",
        "claim_status": "open",
        "paid_amount": 0,
        "description": "Insured vehicle was damaged in a rear-end collision.",
        "state": "TX",
    },
    {
        "claim_id": "CLM-3002",
        "policy_id": "POL-2001",
        "customer_id": "CUST-6002",
        "loss_date": "2024-05-09",
        "reported_date": "2024-05-10",
        "loss_type": "hail_damage_auto",
        "claim_status": "closed",
        "paid_amount": 2100,
        "description": "Vehicle sustained hail dents during a severe storm.",
        "state": "TX",
    },
    {
        "claim_id": "CLM-3003",
        "policy_id": "POL-2001",
        "customer_id": "CUST-6003",
        "loss_date": "2024-07-14",
        "reported_date": "2024-07-15",
        "loss_type": "mechanical_breakdown",
        "claim_status": "denied",
        "paid_amount": 0,
        "description": "Engine failure caused by mechanical breakdown.",
        "state": "TX",
    },
    {
        "claim_id": "CLM-4001",
        "policy_id": "POL-3001",
        "customer_id": "CUST-7001",
        "loss_date": "2024-04-28",
        "reported_date": "2024-04-29",
        "loss_type": "commercial_hail_damage",
        "claim_status": "open",
        "paid_amount": 0,
        "description": "Hail damaged roof of commercial retail building.",
        "state": "TX",
    },
    {
        "claim_id": "CLM-4002",
        "policy_id": "POL-3001",
        "customer_id": "CUST-7002",
        "loss_date": "2024-05-20",
        "reported_date": "2024-05-21",
        "loss_type": "commercial_flood",
        "claim_status": "under_review",
        "paid_amount": 0,
        "description": "Flood water entered warehouse after nearby creek overflowed.",
        "state": "TX",
    },
    {
        "claim_id": "CLM-4003",
        "policy_id": "POL-3001",
        "customer_id": "CUST-7003",
        "loss_date": "2024-02-05",
        "reported_date": "2024-02-05",
        "loss_type": "fire_damage",
        "claim_status": "closed",
        "paid_amount": 8500,
        "description": "Small electrical fire damaged office storage area.",
        "state": "TX",
    },
    {
        "claim_id": "CLM-5001",
        "policy_id": "POL-4001",
        "customer_id": "CUST-8001",
        "loss_date": "2024-08-12",
        "reported_date": "2024-08-13",
        "loss_type": "business_property_theft",
        "claim_status": "open",
        "paid_amount": 0,
        "description": "Retail store reported theft of business equipment after overnight break-in.",
        "state": "TX",
    },
    {
        "claim_id": "CLM-5002",
        "policy_id": "POL-4001",
        "customer_id": "CUST-8002",
        "loss_date": "2024-09-01",
        "reported_date": "2024-09-02",
        "loss_type": "business_interruption",
        "claim_status": "under_review",
        "paid_amount": 0,
        "description": "Business operations suspended for three days after covered fire damage to store area.",
        "state": "TX",
    },
    {
        "claim_id": "CLM-5003",
        "policy_id": "POL-4001",
        "customer_id": "CUST-8003",
        "loss_date": "2024-10-06",
        "reported_date": "2024-10-07",
        "loss_type": "wear_and_tear",
        "claim_status": "denied",
        "paid_amount": 0,
        "description": "Damage resulted from long-term deterioration of business property flooring.",
        "state": "TX",
    },
]

for claim in claims:
    (BASE / "fnol" / f"{claim['claim_id'].lower()}.json").write_text(
        json.dumps(claim, indent=2),
        encoding="utf-8",
    )


# -----------------------------
# Claims history CSV
# -----------------------------
claims_history = pd.DataFrame(claims)
claims_history.to_csv(BASE / "claims_history" / "claims_history.csv", index=False)


# -----------------------------
# Repair estimate CSV
# -----------------------------
repair_estimates = pd.DataFrame(
    [
        ["CLM-2001", "Water extraction", 2500, "Emergency water cleanup"],
        ["CLM-2001", "Ceiling drywall repair", 1800, "Kitchen ceiling damage"],
        ["CLM-2001", "Flooring repair", 3200, "Hardwood flooring water exposure"],
        ["CLM-2002", "Flood cleanup", 4500, "Surface water cleanup"],
        ["CLM-3001", "Rear bumper repair", 1600, "Collision repair"],
        ["CLM-3001", "Paint refinish", 900, "Rear panel paint"],
        ["CLM-3002", "Paintless dent repair", 2100, "Hail damage repair"],
        ["CLM-4001", "Commercial roof repair", 18500, "Hail damage to roof membrane"],
        ["CLM-4002", "Warehouse cleanup", 12000, "Flood cleanup"],
        ["CLM-4003", "Electrical repair", 8500, "Fire-related electrical work"],
        ["CLM-5001", "Equipment replacement", 9500, "Stolen business equipment"],
        ["CLM-5002", "Business income review", 6200, "Three-day interruption estimate"],
    ],
    columns=["claim_id", "item", "estimated_cost", "notes"],
)
repair_estimates.to_csv(BASE / "repair_estimates" / "repair_estimates.csv", index=False)


# -----------------------------
# Adjuster notes TXT
# -----------------------------
for claim in claims:
    note = f"""
Claim ID: {claim['claim_id']}
Policy ID: {claim['policy_id']}
Loss Date: {claim['loss_date']}
Loss Type: {claim['loss_type']}
Claim Status: {claim['claim_status']}

Adjuster Note:
{claim['description']}

Initial Review:
The adjuster should compare the cause of loss against policy coverage, exclusions, deductible, claim history, and supporting documents.
"""
    (BASE / "adjuster_notes" / f"note_{claim['claim_id'].lower()}.txt").write_text(
        note.strip(),
        encoding="utf-8",
    )


# -----------------------------
# Underwriting guidelines TXT
# -----------------------------
guidelines = """
Underwriting Guideline: Property & Casualty Claims Review

General claim review should validate:
- Policy ID
- Claim ID
- Loss date
- Cause of loss
- Applicable coverage
- Deductible
- Exclusions
- Prior claims history
- Supporting documentation

Homeowners:
Sudden and accidental water discharge may be covered.
Flood and surface water are generally excluded unless separate flood coverage exists.
Repeated seepage over 14 days or more is commonly excluded.

Personal Auto:
Collision losses may be covered under collision coverage.
Hail damage may be covered under comprehensive coverage.
Mechanical breakdown is generally excluded.

Commercial Property:
Wind and hail damage may be covered subject to roof condition and deductible.
Flood is generally excluded unless separate flood endorsement exists.
Wear and tear, deterioration, and poor maintenance are generally excluded.

Business Owners Policy:
Theft of business property may be covered if evidence supports forced entry.
Business interruption requires covered direct physical loss.
Wear and tear is excluded.
"""
(BASE / "underwriting_guidelines" / "pc_claims_guideline.txt").write_text(
    guidelines.strip(),
    encoding="utf-8",
)


# -----------------------------
# API data JSON - prior claims
# -----------------------------
prior_claims_api = {
    "source_system": "claims_core_api",
    "generated_at": "2024-11-01T10:30:00Z",
    "records": [
        {
            "customer_id": "CUST-5001",
            "policy_id": "POL-1001",
            "prior_claim_count": 2,
            "prior_claims": ["CLM-2001", "CLM-2002"],
        },
        {
            "customer_id": "CUST-6001",
            "policy_id": "POL-2001",
            "prior_claim_count": 1,
            "prior_claims": ["CLM-3001"],
        },
        {
            "customer_id": "CUST-7001",
            "policy_id": "POL-3001",
            "prior_claim_count": 1,
            "prior_claims": ["CLM-4001"],
        },
        {
            "customer_id": "CUST-8001",
            "policy_id": "POL-4001",
            "prior_claim_count": 1,
            "prior_claims": ["CLM-5001"],
        },
    ],
}

(BASE / "api_data" / "prior_claims_api_response.json").write_text(
    json.dumps(prior_claims_api, indent=2),
    encoding="utf-8",
)


# -----------------------------
# Supporting documents TXT
# -----------------------------
supporting_docs = {
    "inspection_report_clm_2001.txt": """
Claim ID: CLM-2001
Inspection Report:
Visible water staining observed on kitchen ceiling.
Moisture readings elevated near ceiling and flooring below upstairs bathroom.
No evidence of flood water, storm surge, or long-term seepage.
Cause appears consistent with sudden plumbing failure.
""",
    "photo_summary_clm_2001.txt": """
Claim ID: CLM-2001
Photo Summary:
Photos show ceiling discoloration, fallen drywall pieces, and wet hardwood flooring.
Images support water discharge from upstairs bathroom area.
""",
    "inspection_report_clm_4001.txt": """
Claim ID: CLM-4001
Inspection Report:
Commercial roof shows hail impact marks and membrane damage.
No evidence of poor maintenance at initial inspection.
Roof condition appears consistent with reported storm event.
""",
    "police_report_clm_5001.txt": """
Claim ID: CLM-5001
Police Report Summary:
Break-in reported at insured retail store.
Rear entrance showed signs of forced entry.
Missing items include point-of-sale equipment, laptop, and inventory scanner.
""",
}

for filename, content in supporting_docs.items():
    (BASE / "supporting_documents" / filename).write_text(
        content.strip(),
        encoding="utf-8",
    )


print("Enterprise-style P&C sample dataset created successfully.")
print(f"Policies: {len(policies)}")
print(f"Claims: {len(claims)}")
print("Data formats included: TXT, JSON, CSV")