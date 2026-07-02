"""
Cleans the raw SHL catalog scrape into a structured file the retrieval
and agent layers consume.

Why this step exists (for the approach doc):
- The raw scrape includes 7 "Job Solution" bundle products (e.g. "Entry
  Level Sales Solution"). The assignment explicitly restricts scope to
  Individual Test Solutions, so these are dropped here, once, rather than
  filtered ad-hoc at query time.
- SHL's public catalog UI uses single-letter test-type codes (A, B, C, D,
  E, K, P, S) next to each product. The raw scrape only gives us the full
  category names under "keys" (e.g. "Personality & Behavior"). We map
  those to the standard codes so our /chat response can include
  "test_type" as specified in the API contract.
"""
import json
import re

RAW_PATH = "raw_catalog.json"
OUT_PATH = "catalog_clean.json"

# Standard SHL test-type taxonomy (category name -> single-letter code)
CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# Job-solution bundles identified by inspecting the raw scrape: their
# product-catalog URL slug ends in "-solution". Individual tests never do.
JOB_SOLUTION_SLUG_SUFFIX = "-solution/"


def load_raw():
    with open(RAW_PATH, encoding="utf-8") as f:
        text = f.read()
    # raw scrape has a couple of stray control characters in description
    # fields (line breaks captured verbatim from the site) -> strict=False
    return json.loads(text, strict=False)


def is_job_solution(item):
    return item["link"].rstrip().endswith(JOB_SOLUTION_SLUG_SUFFIX)


def clean_text(t):
    if not t:
        return ""
    # collapse whitespace/newlines that leaked in from scraping
    return re.sub(r"\s+", " ", t).strip()


def main():
    raw = load_raw()
    cleaned = []
    dropped_job_solutions = []

    for item in raw:
        if is_job_solution(item):
            dropped_job_solutions.append(item["name"])
            continue

        keys = item.get("keys", [])
        test_types = sorted({CATEGORY_TO_CODE.get(k, "") for k in keys} - {""})

        cleaned.append({
            "id": item["entity_id"],
            "name": item["name"].strip(),
            "url": item["link"].strip(),
            "description": clean_text(item.get("description", "")),
            "test_types": test_types,          # e.g. ["P"]
            "categories": keys,                 # e.g. ["Personality & Behavior"]
            "job_levels": item.get("job_levels", []),
            "languages": item.get("languages", []),
            "duration": clean_text(item.get("duration", "")),
            "remote_testing": item.get("remote", "") == "yes",
            "adaptive_irt": item.get("adaptive", "") == "yes",
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Raw items:            {len(raw)}")
    print(f"Dropped job solutions: {len(dropped_job_solutions)} -> {dropped_job_solutions}")
    print(f"Clean individual tests: {len(cleaned)}")
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
