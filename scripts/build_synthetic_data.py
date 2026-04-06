#!/usr/bin/env python3
"""Generate synthetic paired corpus (v0/v1) and queries.jsonl under data/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from adaptive_rag.chunking import load_corpus_dir  # noqa: E402

# Must match configs/local.yaml chunking
MAX_WORDS = 50
OVERLAP = 10


def policy_a_v0() -> str:
    return """# Commercial Property Policy — Form CP-2024-A

## 1. Insuring Agreement
We cover direct physical loss to covered property at the described premises caused by a covered peril.

## 2. Covered Property
Covered property includes the building, business personal property, and outdoor property listed in the declarations.

## 3. Windstorm and Hail
Windstorm and hail are covered perils. The deductible for windstorm and hail losses is **$1,000** per occurrence unless amended by endorsement.

## 4. Flood
Flood is excluded unless Flood Endorsement FE-12 is attached and premium is paid.

## 5. Duties After Loss
The insured must notify us promptly, protect property from further damage, and cooperate in the investigation.

## 6. Cancellation
We may cancel for nonpayment with 10 days notice.
"""


def policy_a_v1() -> str:
    return policy_a_v0().replace(
        "The deductible for windstorm and hail losses is **$1,000** per occurrence unless amended by endorsement.",
        "The deductible for windstorm and hail losses is **$5,000** per occurrence unless amended by endorsement.",
    ).replace(
        "We may cancel for nonpayment with 10 days notice.",
        "We may cancel for nonpayment with 14 days notice.",
    )


def policy_b_v0() -> str:
    return """# General Liability — GL-STD-101

## A. Coverage A — Bodily Injury
We pay amounts the insured becomes legally obligated to pay because of bodily injury.

## B. Coverage B — Property Damage
We pay amounts because of property damage to third-party property.

## C. Exclusions
Expected or intended injury is excluded. **Cyber incidents** are excluded except as provided in Endorsement CY-1 if attached.

## D. Limits
Each occurrence limit is shown in the declarations. The aggregate limit applies separately to products/completed operations.
"""


def policy_b_v1() -> str:
    return policy_b_v0().replace(
        "**Cyber incidents** are excluded except as provided in Endorsement CY-1 if attached.",
        "**Cyber incidents** are covered when caused by a listed network security failure, subject to a **$250,000** sublimit.",
    )


def policy_c_v0() -> str:
    return """# Workers Compensation and Employers Liability — WC-50

## 1. Workers Compensation
We pay statutory workers compensation benefits where applicable.

## 2. Employers Liability
Employers liability applies as described in Part Two.

## 3. Premium Audit
Premium is estimated at inception and subject to audit. Employers must maintain payroll records for **3 years** after policy expiration.

## 4. Subrogation
We may exercise rights of subrogation against third parties responsible for injury.
"""


def policy_c_v1() -> str:
    return policy_c_v0().replace(
        "Employers must maintain payroll records for **3 years** after policy expiration.",
        "Employers must maintain payroll records for **5 years** after policy expiration.",
    )


def find_chunk_index(chunks, doc_id: str, needle: str) -> int:
    for c in chunks:
        if c.doc_id == doc_id and needle.lower() in c.text.lower():
            return c.chunk_index
    raise ValueError(f"needle not found: {doc_id!r} {needle!r}")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "corpus" / "v0").mkdir(parents=True, exist_ok=True)
    (DATA / "corpus" / "v1").mkdir(parents=True, exist_ok=True)

    (DATA / "corpus" / "v0" / "policy_a.md").write_text(policy_a_v0(), encoding="utf-8")
    (DATA / "corpus" / "v1" / "policy_a.md").write_text(policy_a_v1(), encoding="utf-8")
    (DATA / "corpus" / "v0" / "policy_b.md").write_text(policy_b_v0(), encoding="utf-8")
    (DATA / "corpus" / "v1" / "policy_b.md").write_text(policy_b_v1(), encoding="utf-8")
    (DATA / "corpus" / "v0" / "policy_c.md").write_text(policy_c_v0(), encoding="utf-8")
    (DATA / "corpus" / "v1" / "policy_c.md").write_text(policy_c_v1(), encoding="utf-8")

    chunks_v1 = load_corpus_dir("corpus/v1/*.md", "v1", MAX_WORDS, OVERLAP, DATA)

    specs: list[dict[str, str]] = [
        {"id": "q01", "cat": "agent", "doc": "policy_a", "needle": "Insuring Agreement", "q": "What does the commercial property policy cover at a high level?", "a": "The policy covers direct physical loss to covered property at the described premises caused by a covered peril."},
        {"id": "q02", "cat": "agent", "doc": "policy_a", "needle": "Covered property includes", "q": "What property is included as covered property?", "a": "Covered property includes the building, business personal property, and outdoor property listed in the declarations."},
        {"id": "q03", "cat": "agent", "doc": "policy_a", "needle": "$5,000", "q": "What is the windstorm and hail deductible?", "a": "The deductible for windstorm and hail losses is $5,000 per occurrence unless amended by endorsement."},
        {"id": "q04", "cat": "agent", "doc": "policy_a", "needle": "Flood is excluded", "q": "Is flood covered under the base policy?", "a": "Flood is excluded unless Flood Endorsement FE-12 is attached and premium is paid."},
        {"id": "q05", "cat": "agent", "doc": "policy_b", "needle": "Coverage A", "q": "What does Coverage A pay for?", "a": "We pay amounts the insured becomes legally obligated to pay because of bodily injury."},
        {"id": "q06", "cat": "agent", "doc": "policy_b", "needle": "Coverage B", "q": "What does Coverage B address?", "a": "We pay amounts because of property damage to third-party property."},
        {"id": "q07", "cat": "agent", "doc": "policy_b", "needle": "$250,000", "q": "How are cyber incidents treated in general liability?", "a": "Cyber incidents are covered when caused by a listed network security failure, subject to a $250,000 sublimit."},
        {"id": "q08", "cat": "agent", "doc": "policy_c", "needle": "5 years", "q": "How long must payroll records be maintained after policy expiration?", "a": "Employers must maintain payroll records for 5 years after policy expiration."},
        {"id": "q09", "cat": "compliance", "doc": "policy_a", "needle": "$5,000", "q": "State the post-amendment wind/hail deductible language.", "a": "The deductible for windstorm and hail losses is $5,000 per occurrence unless amended by endorsement."},
        {"id": "q10", "cat": "compliance", "doc": "policy_a", "needle": "14 days", "q": "What cancellation notice period applies for nonpayment?", "a": "We may cancel for nonpayment with 14 days notice."},
        {"id": "q11", "cat": "compliance", "doc": "policy_b", "needle": "$250,000", "q": "Provide the cyber coverage rule after amendment.", "a": "Cyber incidents are covered when caused by a listed network security failure, subject to a $250,000 sublimit."},
        {"id": "q12", "cat": "compliance", "doc": "policy_c", "needle": "5 years", "q": "What record retention period applies to payroll records after expiration?", "a": "Employers must maintain payroll records for 5 years after policy expiration."},
        {"id": "q13", "cat": "compliance", "doc": "policy_a", "needle": "Duties After Loss", "q": "List duties after loss.", "a": "The insured must notify us promptly, protect property from further damage, and cooperate in the investigation."},
        {"id": "q14", "cat": "compliance", "doc": "policy_b", "needle": "aggregate limit", "q": "How do aggregate limits apply?", "a": "The aggregate limit applies separately to products/completed operations."},
        {"id": "q15", "cat": "compliance", "doc": "policy_c", "needle": "statutory workers compensation", "q": "What statutory benefit is paid?", "a": "We pay statutory workers compensation benefits where applicable."},
        {"id": "q16", "cat": "compliance", "doc": "policy_c", "needle": "subrogation", "q": "What subrogation rights may be exercised?", "a": "We may exercise rights of subrogation against third parties responsible for injury."},
        {"id": "q17", "cat": "customer", "doc": "policy_a", "needle": "$5,000", "q": "If a windstorm damages my roof, what deductible applies?", "a": "The deductible for windstorm and hail losses is $5,000 per occurrence unless amended by endorsement."},
        {"id": "q18", "cat": "customer", "doc": "policy_a", "needle": "Flood is excluded", "q": "Does this policy cover flood damage automatically?", "a": "Flood is excluded unless Flood Endorsement FE-12 is attached and premium is paid."},
        {"id": "q19", "cat": "customer", "doc": "policy_b", "needle": "network security failure", "q": "Is a ransomware-related loss covered under GL cyber wording?", "a": "Cyber incidents are covered when caused by a listed network security failure, subject to a $250,000 sublimit."},
        {"id": "q20", "cat": "customer", "doc": "policy_c", "needle": "5 years", "q": "How long should I keep payroll records?", "a": "Employers must maintain payroll records for 5 years after policy expiration."},
        {"id": "q21", "cat": "customer", "doc": "policy_a", "needle": "Covered property includes", "q": "What kinds of property are covered?", "a": "Covered property includes the building, business personal property, and outdoor property listed in the declarations."},
        {"id": "q22", "cat": "customer", "doc": "policy_b", "needle": "third-party property", "q": "Does Coverage B cover damage to someone else's property?", "a": "We pay amounts because of property damage to third-party property."},
        {"id": "q23", "cat": "customer", "doc": "policy_c", "needle": "Part Two", "q": "What is Part Two about?", "a": "Employers liability applies as described in Part Two."},
        {"id": "q24", "cat": "customer", "doc": "policy_a", "needle": "14 days", "q": "How much notice do I get if the insurer cancels for nonpayment?", "a": "We may cancel for nonpayment with 14 days notice."},
    ]

    lines = []
    for s in specs:
        idx = find_chunk_index(chunks_v1, s["doc"], s["needle"])
        row = {
            "id": s["id"],
            "text": s["q"],
            "category": s["cat"],
            "gold_doc": s["doc"],
            "gold_chunk_index": idx,
            "gold_answer_v1": s["a"],
            "requires_v1_post_amendment": True,
        }
        lines.append(json.dumps(row))

    (DATA / "queries.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote corpus + queries; chunk params:", MAX_WORDS, OVERLAP)


if __name__ == "__main__":
    main()
