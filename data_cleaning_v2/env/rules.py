"""
rules.py — Business rules and schema constraints.

Every cleaning action is validated against these rules.
The agent must satisfy ALL rules for a row to be "clean".
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional


SCHEMA = {
    "name":         {"type": str, "required": True},
    "email":        {"type": str, "required": True},
    "phone":        {"type": str, "required": True},
    "date_of_birth":{"type": str, "required": True},
    "city":         {"type": str, "required": True},
    "signup_date":  {"type": str, "required": True},
    "status":       {"type": str, "required": True},
}

VALID_STATUSES = {"Active", "Inactive", "Pending"}

VALID_CITIES = {
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat"
}


def is_valid_email(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value))


def is_valid_phone(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return len(re.sub(r'\D', '', value)) == 10


def normalize_phone(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    digits = re.sub(r'\D', '', value)
    return digits if len(digits) == 10 else None


def is_valid_date(value: Any) -> bool:
    """Date must be YYYY-MM-DD. Year range: 1900 to current year (dynamic)."""
    if not isinstance(value, str):
        return False
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', value):
        return False
    try:
        from datetime import date
        y, m, d = (int(x) for x in value.split('-'))
        date(y, m, d)  # raises ValueError for invalid dates like Feb 30
        return 1900 <= y <= datetime.now().year
    except (ValueError, Exception):
        return False


def normalize_date(value: Any) -> Optional[str]:
    """Convert DD/MM/YYYY or DD-MM-YYYY to YYYY-MM-DD. Returns None if unparseable."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if is_valid_date(value):
        return value
    # DD/MM/YYYY
    m = re.match(r'^(\d{2})/(\d{2})/(\d{4})$', value)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    # DD-MM-YYYY
    m = re.match(r'^(\d{2})-(\d{2})-(\d{4})$', value)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


def is_valid_name(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    return value == value.title() and bool(re.match(r'^[A-Za-z\s]+$', value))


def normalize_name(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value.strip().title()


def is_valid_status(value: Any) -> bool:
    return value in VALID_STATUSES


def normalize_status(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    canonical = value.strip().capitalize()
    if canonical in VALID_STATUSES:
        return canonical
    # Handle ALL-CAPS like "INACTIVE"
    titled = value.strip().title()
    return titled if titled in VALID_STATUSES else None


def validate_row(row: Dict[str, Any]) -> Dict[str, List[str]]:
    """Returns dict of field -> error list. Empty = clean row."""
    errors: Dict[str, List[str]] = {}

    def add(field, msg):
        errors.setdefault(field, []).append(msg)

    if not row.get("name"):
        add("name", "missing")
    elif not is_valid_name(row["name"]):
        add("name", f"not title case: '{row['name']}'")

    if not row.get("email"):
        add("email", "missing")
    elif not is_valid_email(row["email"]):
        add("email", f"invalid format: '{row['email']}'")

    if not row.get("phone"):
        add("phone", "missing")
    elif not is_valid_phone(row["phone"]):
        add("phone", f"not 10 digits: '{row['phone']}'")
    elif row["phone"] != re.sub(r'\D', '', str(row["phone"])):
        add("phone", f"contains non-digit characters: '{row['phone']}'")

    if not row.get("date_of_birth"):
        add("date_of_birth", "missing")
    elif not is_valid_date(row["date_of_birth"]):
        add("date_of_birth", f"wrong format (need YYYY-MM-DD): '{row['date_of_birth']}'")

    if not row.get("city"):
        add("city", "missing")
    elif row["city"] != row["city"].strip().title():
        add("city", f"not title case: '{row['city']}'")

    if not row.get("signup_date"):
        add("signup_date", "missing")
    elif not is_valid_date(row["signup_date"]):
        add("signup_date", f"wrong format (need YYYY-MM-DD): '{row['signup_date']}'")

    if not row.get("status"):
        add("status", "missing")
    elif not is_valid_status(row["status"]):
        add("status", f"invalid value: '{row['status']}'")

    return errors


def is_outlier(row: Dict[str, Any]) -> Dict[str, str]:
    outliers = {}
    current_year = datetime.now().year
    dob = row.get("date_of_birth", "")
    if is_valid_date(dob):
        age = current_year - int(dob.split("-")[0])
        if age < 18:
            outliers["date_of_birth"] = f"Age {age} too young (<18)"
        elif age > 100:
            outliers["date_of_birth"] = f"Age {age} too old (>100)"
    sig = row.get("signup_date", "")
    if is_valid_date(sig):
        if int(sig.split("-")[0]) > current_year:
            outliers["signup_date"] = f"Signup year {sig.split('-')[0]} is in the future"
    return outliers


def count_errors(row: Dict[str, Any]) -> int:
    return sum(len(v) for v in validate_row(row).values())
