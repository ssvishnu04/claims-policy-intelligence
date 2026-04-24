from pathlib import Path
import json
import pandas as pd

BASE = Path("data/raw")

folders = [
    "policies",
    "fnol",
    "adjuster_notes",
    "underwriting_guidelines",
    "claims_history",
    "repair_estimates",
    "supporting_documents",
    "api_data",
]

for folder in folders:
    (BASE / folder).mkdir(parents=True, exist_ok=True)


policies = {
    "homeowners_policy_pol_1001.txt": """
Policy ID: POL-1001
Product: Homeowners HO-3
State: Texas

WATER DAMAGE
We cover sudden and accidental discharge or overflow of water from within a plumbing system.

EXCLUSIONS
We do not cover flood, surface water, underground water seepage, or repeated leakage over 14 days or more.

DEDUCTIBLE
A deductible of $2,500 applies to covered property losses.
""",
    "auto_policy_pol_2001.txt": """
Policy ID: POL-2001
Product: Personal Auto
State: Texas

COLLISION COVERAGE
We cover direct and accidental loss to your covered auto caused by collision.

COMPREHENSIVE COVERAGE
We cover theft, fire, vandalism, hail, falling objects, and animal impact.

EXCLUSIONS
We do not cover intentional damage, racing, mechanical breakdown, or wear and tear.

DEDUCTIBLE
A deductible of $500 applies to collision losses.
""",
    "commercial_property_policy_pol_3001.txt": """
Policy ID: POL-3001
Product: Commercial Property
State: Texas

PROPERTY COVERAGE
We cover direct physical loss to covered commercial property caused by covered causes of loss.

WIND AND HAIL
Wind and hail damage may be covered subject to deductible and policy conditions.

EXCLUSIONS
We do not cover flood, earth movement, wear and tear, or poor maintenance.

DEDUCTIBLE
A deductible of $5,000 applies to covered commercial property losses.
""",
}

for filename, content in policies.items():
    (BASE / "policies" / filename).write_text(content.strip(), encoding="utf-8")


claims = [
    {
        "claim_id": "CLM-2001",
        "policy_id": "POL-1001",
        "loss_type": "water_damage",
        "description": "Sudden pipe burst in upstairs bathroom caused water damage to kitchen ceiling.",
        "status": "open",
        "state": "TX",
    },
    {
        "claim_id": "CLM-2002",
        "policy_id": "POL-1001",
        "loss_type": "flood_damage",
        "description": "Heavy rain caused surface water to enter the home through the front door.",
        "status": "open",
        "state": "TX",
    },
    {
        "claim_id": "CLM-2003",
        "policy_id": "POL-1001",
        "loss_type": "roof_wind_damage",
        "description": "Windstorm damaged roof shingles on residential property.",
        "status": "closed",
        "state": "TX",
    },
    {
        "claim_id": "CLM-3001",
        "policy_id": "POL-2001",
        "loss_type": "auto_collision",
        "description": "Insured vehicle was damaged in a rear-end collision.",
        "status": "open",
        "state": "TX",
    },
    {
        "claim_id": "CLM-3002",
        "policy_id": "POL-2001",
        "loss_type": "hail_damage_auto",
        "description": "Vehicle sustained hail dents during storm.",
        "status": "open",
        "state": "TX",
    },
    {
        "claim_id": "CLM-3003",
        "policy_id": "POL-2001",
        "loss_type": "mechanical_breakdown",
        "description": "Engine failed due to mechanical breakdown.",
        "status": "closed",
        "state": "TX",
    },
    {
        "claim_id": "CLM-4001",
        "policy_id": "POL-3001",
        "loss_type": "commercial_hail_damage",
        "description": "Hail damaged roof of commercial retail building.",
        "status": "open",
        "state": "TX",
    },
    {
        "claim_id": "CLM-4002",
        "policy_id": "POL-3001",
        "loss_type": "commercial_flood",
        "description": "Flood water entered warehouse after nearby creek overflowed.",
        "status": "open",
        "state": "TX",
    },
    {
        "claim_id": "CLM-4003",
        "policy_id": "POL-3001",
        "loss_type": "fire_damage",
        "description": "Small electrical fire damaged office storage area.",
        "status": "closed",
        "state": "TX",
    },
    {
        "claim_id": "CLM-4004",
        "policy_id": "POL-3001",
        "loss_type": "wear_and_tear",
        "description": "Roof leak caused by long-term deterioration and poor maintenance.",
        "status": "closed",
        "state": "TX",
    },
]

for claim in claims:
    (BASE / "fnol" / f"{claim['claim_id'].lower()}.json").write_text(
        json.dumps(claim, indent=2),
        encoding="utf-8",
    )

    note = f"""
Claim ID: {claim['claim_id']}
Policy ID: {claim['policy_id']}
Loss Type: {claim['loss_type']}

Adjuster Note:
{claim['description']}

Initial review indicates the claim should be compared against policy coverage, exclusions, deductible, and loss conditions.
"""
    (BASE / "adjuster_notes" / f"note_{claim['claim_id'].lower()}.txt").write_text(
        note.strip(),
        encoding="utf-8",
    )


claims_history = pd.DataFrame(claims)
claims_history["paid_amount"] = [0, 0, 4200, 0, 2100, 0, 0, 0, 8500, 0]
claims_history.to_csv(BASE / "claims_history" / "claims_history.csv", index=False)


repair_estimates = pd.DataFrame(
    [
        ["CLM-2001", "Ceiling drywall repair", 1800],
        ["CLM-2001", "Flooring repair", 3200],
        ["CLM-2002", "Water extraction", 4500],
        ["CLM-3001", "Rear bumper repair", 1600],
        ["CLM-3002", "Paintless dent repair", 2100],
        ["CLM-4001", "Commercial roof repair", 18500],
        ["CLM-4002", "Warehouse cleanup", 12000],
        ["CLM-4003", "Electrical repair", 8500],
    ],
    columns=["claim_id", "item", "estimated_cost"],
)
repair_estimates.to_csv(BASE / "repair_estimates" / "repair_estimates.csv", index=False)


uw_guideline = """
Underwriting Guideline: P&C Claims Review

Adjusters should validate:
- Policy ID
- Claim ID
- Cause of loss
- Loss date
- Applicable exclusions
- Deductible
- Prior claims history
- Supporting documentation

Water damage caused by sudden pipe burst may be covered under homeowners policies.
Flood and surface water are commonly excluded unless separate flood coverage exists.
Auto collision may be covered if collision coverage is active.
Mechanical breakdown is commonly excluded under personal auto policies.
Commercial wind and hail losses may be covered subject to deductible and maintenance review.
"""
(BASE / "underwriting_guidelines" / "pc_claims_guideline.txt").write_text(
    uw_guideline.strip(),
    encoding="utf-8",
)

print("Sample P&C insurance dataset created successfully.")