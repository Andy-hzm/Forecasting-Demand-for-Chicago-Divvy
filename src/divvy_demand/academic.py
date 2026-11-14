"""
University academic calendar ETL — curated term boundaries, no API.

processed/events/academic   one row per (school, term)
    school       str   # 'uchicago', 'northwestern', 'uic', 'depaul', 'loyola', 'iit', 'columbia'
    term         str   # e.g. 'autumn_2023'
    start, end   datetime   # first day of classes, last day of finals
    lat, lng     float
    system       str   # 'quarter' | 'semester'
    is_remote    int   # 1 if instruction was (mostly) remote that term — 2020-21 pandemic terms
    note         str

Use divvy_demand.features.add_university_session() to turn this into per-station
daily flags (nearest school in session / term week / finals).
"""
import logging

import pandas as pd

from divvy_demand.infra.s3_client import S3Client
from divvy_demand.calendars.academic_terms import ACADEMIC_TERMS, SCHOOLS

logger = logging.getLogger(__name__)


def build_academic_terms() -> pd.DataFrame:
    if not ACADEMIC_TERMS:
        logger.warning("ACADEMIC_TERMS is empty — calendars/academic_terms.py not populated yet")
        return pd.DataFrame(columns=["school", "term", "start", "end", "lat", "lng", "system", "is_remote", "note"])
    t = pd.DataFrame(ACADEMIC_TERMS)
    t["start"] = pd.to_datetime(t["start"])
    t["end"] = pd.to_datetime(t["end"])
    t["lat"] = t["school"].map(lambda s: SCHOOLS[s]["lat"])
    t["lng"] = t["school"].map(lambda s: SCHOOLS[s]["lng"])
    t["system"] = t["school"].map(lambda s: SCHOOLS[s]["system"])
    t["note"] = t.get("note", "").fillna("")
    t["is_remote"] = t["note"].str.contains("remote", case=False).astype(int)
    bad = t[t["end"] <= t["start"]]
    if len(bad):
        raise ValueError(f"Terms with end <= start:\n{bad}")
    return (
        t[["school", "term", "start", "end", "lat", "lng", "system", "is_remote", "note"]]
        .sort_values(["school", "start"]).reset_index(drop=True)
    )


def build_academic(s3: S3Client) -> pd.DataFrame:
    terms = build_academic_terms()
    s3.write_processed(terms, "events/academic")
    logger.info(f"Academic: {len(terms)} terms across {terms['school'].nunique()} schools")
    return terms
