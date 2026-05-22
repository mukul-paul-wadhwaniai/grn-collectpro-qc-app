import os
import re
import math
import json
import secrets
import sys
import threading
from collections import defaultdict

import pandas as pd
from pathlib import Path
from flask import (
    Flask, render_template, jsonify, request,
    send_from_directory, session, redirect, url_for,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from util import ensure_s3_image_cached, parse_s3_url

from db import (
    init_db, save_review, delete_review, get_reviews_for_sample,
    get_reviews_all_teams_for_sample,
    get_all_reviews, get_issue_types, add_issue_type,
    get_review_counts_by_sample_and_team,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_FILE = "/data/temp_dir/latest.parquet"
SAMPLES_DASHBOARD_FILE = os.environ.get(
    "SAMPLES_DASHBOARD_FILE",
    "/data/temp_dir/samples_dashboard.parquet",
)
ADDITIONAL_METADATA_FILE = os.environ.get(
    "ADDITIONAL_METADATA_FILE",
    "/data/temp_dir/additional_metadata.parquet",
)
STATIC_IMG_DIR = Path("static/images")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "agri-grn-prod-dip-bucket")

# Optional unified admin view (read-only across all teams). Set GRAIN_REVIEW_ADMIN_PASSWORD to enable login user "admin".
_GRAIN_ADMIN_PASSWORD = os.environ.get("GRAIN_REVIEW_ADMIN_PASSWORD", "").strip()

# Team accounts: username → { password, team }
TEAM_ACCOUNTS = {
    "ml": {"password": r"qM565~v-Tw\K", "team": "ML"},
    "programmes": {"password": r"qM565~v-Tw\K", "team": "Programmes"},
    "product": {"password": r"qM565~v-Tw\K", "team": "Product"},
}
if _GRAIN_ADMIN_PASSWORD:
    TEAM_ACCOUNTS["admin"] = {
        "password": _GRAIN_ADMIN_PASSWORD,
        "team": "Admin",
        "is_admin": True,
    }
TEAMS = ["ML", "Programmes", "Product"]

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
# app.config["TEMPLATES_AUTO_RELOAD"] = True

STATIC_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Form Types
MIXED_FORM = "MixedSample_FineProtocol"
CATEGORY_FORM = "CategorySample_FineProtocol"
AMBIGUOUS_FORM = "CategorySample_Ambiguous_FineProtocol"

# Additional Metadata form titles are listed in prepare_data.EXTRA_FORM_NAMES.
# The review UI loads submissions from additional_metadata.parquet (built by prepare_data).
# When no form exists for a sample that is in the main review parquet, the API still
# returns one row so reviewers can record accept/flag against this sentinel datapoint_id.
DEFAULT_ADDITIONAL_METADATA_FORM_LABEL = "Additional metadata fine protocol"
ADDITIONAL_METADATA_ABSENT_DP_PREFIX = "__no_additional_metadata__"


def _additional_metadata_absent_datapoint_id(sample_id: int) -> str:
    return f"{ADDITIONAL_METADATA_ABSENT_DP_PREFIX}-{int(sample_id)}"


# ---------------------------------------------------------------------------
# Fine Protocol Category Config
# ---------------------------------------------------------------------------
FINE_PROTOCOL_STANDARD = {
    "obvious_good_grains": "GG",
    "obvious_good_split_grains": "GG",
    "purple_grains": "GG",
    "soiled_grains": "GG",
    "obvious_damaged_grains": "DG",
    "obvious_green_grains": "GrG",
    "mud_balls": "FM",
    "wood": "FM",
    "peels": "FM",
    "small_particles": "FM",
    "bengal_gram": "FM",
    "wheat": "FM",
    "pigeon_pea": "FM",
    "sorghum": "FM",
    "maize": "FM",
    "black_gram": "FM",
    "green_gram": "FM",
    "other": "FM",
    "small_good_grains": "SG",
    "small_split_grains": "SG",
}

FINE_PROTOCOL_AMBIGUOUS = {
    "spotted_grains": "GG",
    "shrunken_good_grains": "GG",
    "dark_grains": "DG",
    "white_grains": "DG",
    "subjective_damaged_grains": "DG",
    "possibly_good_green_grains": "GrG",
    "possibly_damaged_green_grains": "GrG",
}

ALL_CATEGORIES = {**FINE_PROTOCOL_STANDARD, **FINE_PROTOCOL_AMBIGUOUS}

SUPERCAT_ORDER = ["GG", "DG", "GrG", "FM", "SG"]
SUPERCAT_LABELS = {
    "GG": "Good Grains",
    "DG": "Damaged Grains",
    "GrG": "Green Grains",
    "FM": "Foreign Matter",
    "SG": "Small Grains (< 4mm)",
}

MIXED_EXPECTED_REFS = ["absent", "a4_sheet", "50_note", "100_note"]

SUPERCAT_GROUPS = defaultdict(list)
for _cat, _sc in ALL_CATEGORIES.items():
    SUPERCAT_GROUPS[_sc].append(_cat)


# ---------------------------------------------------------------------------
# Pre-fetch images (S3 → VM local cache)
# ---------------------------------------------------------------------------
def pre_download_images(dataframe):
    """Download missing images from S3 into static/images for the review UI."""
    print(f"Checking and pre-fetching images from S3 (bucket={S3_BUCKET_NAME}) …")
    STATIC_IMG_DIR.mkdir(parents=True, exist_ok=True)
    local_paths = []
    n_downloaded = 0
    for _, row in dataframe.iterrows():
        s3_url = row.get("s3_url")
        if not s3_url or pd.isna(s3_url):
            local_paths.append(None)
            continue
        _, _, filename = parse_s3_url(str(s3_url))
        local_filepath = STATIC_IMG_DIR / filename
        existed = local_filepath.exists() and local_filepath.stat().st_size > 0
        cached = ensure_s3_image_cached(
            s3_url=str(s3_url),
            cache_dir=STATIC_IMG_DIR,
            bucket_name=S3_BUCKET_NAME,
        )
        if cached and not existed:
            n_downloaded += 1
        if cached and cached.exists():
            local_paths.append(f"/static/images/{filename}")
        else:
            local_paths.append(None)
    dataframe["local_image_path"] = local_paths
    print(f"S3 image sync complete ({n_downloaded} newly downloaded).")
    return dataframe


# ---------------------------------------------------------------------------
# NaN / None helper
# ---------------------------------------------------------------------------
def sanitize(val):
    """Convert NaN / None to Python None for JSON safety."""
    if pd.isna(val):
        return None
    return val


def _display_label(key: str) -> str:
    s = str(key).replace("_", " ").strip()
    if not s:
        return str(key)
    return s[0].upper() + s[1:] if len(s) > 1 else s.upper()


def _flatten_form_payload(data, prefix: str = "", max_depth: int = 12) -> list[dict]:
    """Nested form `data` JSON → flat {label, value} rows for the review UI."""
    rows: list[dict] = []

    def scalar_display(v) -> str:
        if v is None:
            return "—"
        try:
            if pd.isna(v):
                return "—"
        except (TypeError, ValueError):
            pass
        return str(v)[:4000]

    if max_depth <= 0:
        return [{"label": prefix or "value", "value": scalar_display(data)}]

    if isinstance(data, dict):
        for k, v in data.items():
            lab = f"{prefix} › {_display_label(k)}" if prefix else _display_label(str(k))
            if isinstance(v, dict):
                rows.extend(_flatten_form_payload(v, lab, max_depth - 1))
            elif isinstance(v, list):
                try:
                    s = json.dumps(v, default=str)
                except TypeError:
                    s = str(v)
                rows.append({"label": lab, "value": s[:4000]})
            else:
                rows.append({"label": lab, "value": scalar_display(v)})
    else:
        rows.append({"label": prefix or "Value", "value": scalar_display(data)})
    return rows


def _refresh_sample_numbers_union():
    """Merge sample IDs from main parquet + additional-metadata export for the dropdown."""
    global SAMPLE_NUMBERS
    if df is None or df.empty:
        base: list[int] = []
    else:
        base = sorted(df["sample_number"].dropna().unique().astype(int).tolist())
    if additional_metadata_df is not None and not additional_metadata_df.empty:
        extra = additional_metadata_df["sample_number"].dropna().unique().astype(int).tolist()
        SAMPLE_NUMBERS = sorted(set(base) | set(int(x) for x in extra))
    else:
        SAMPLE_NUMBERS = base


# ---------------------------------------------------------------------------
# Auto-sync: reload parquet when the file changes on disk
# ---------------------------------------------------------------------------
_data_lock = threading.Lock()
_parquet_mtime: float = 0.0
_samples_dashboard_mtime: float = 0.0
_additional_metadata_mtime: float = 0.0
df: pd.DataFrame | None = None
samples_dashboard_df: pd.DataFrame | None = None
additional_metadata_df: pd.DataFrame | None = None
SAMPLE_NUMBERS: list[int] = []


def _load_data():
    """Load parquet + pre-fetch images, then swap global state.

    On failure the previous df / SAMPLE_NUMBERS stay in place so the app
    keeps serving stale-but-valid data.  _parquet_mtime is still updated
    to avoid retrying on every subsequent request (the next pipeline run
    will produce a new file with a different mtime).
    """
    global df, SAMPLE_NUMBERS, _parquet_mtime
    try:
        print(f"Loading data from {DATA_FILE} …")
        new_df = pd.read_parquet(DATA_FILE)
        new_df = pre_download_images(new_df)
        new_mtime = os.path.getmtime(DATA_FILE)
        # Atomic swap — only reached if everything above succeeded
        df = new_df
        _parquet_mtime = new_mtime
        print(f"✅ Loaded {len(df)} rows from main parquet")
    except Exception as e:
        print(f"⚠️ RELOAD FAILED: {e} — continuing with previous data")
        import traceback
        traceback.print_exc()
        # Update mtime so we don't retry on every request.
        # Next pipeline run will produce a new file with a different mtime.
        try:
            _parquet_mtime = os.path.getmtime(DATA_FILE)
        except OSError:
            pass
    _reload_samples_dashboard()
    _reload_additional_metadata()
    _refresh_sample_numbers_union()
    try:
        print(f"   → {len(SAMPLE_NUMBERS)} samples in dropdown (incl. additional-metadata-only)")
    except Exception:
        pass


def _mtime_or_zero(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _reload_additional_metadata():
    """Load additional_metadata.parquet (Additional Metadata forms) for the review UI."""
    global additional_metadata_df, _additional_metadata_mtime
    try:
        additional_metadata_df = pd.read_parquet(ADDITIONAL_METADATA_FILE)
        _additional_metadata_mtime = _mtime_or_zero(ADDITIONAL_METADATA_FILE)
        print(
            f"✅ Loaded additional metadata ({len(additional_metadata_df)} rows) "
            f"from {ADDITIONAL_METADATA_FILE}"
        )
    except Exception as e:
        print(f"⚠️ Additional metadata load skipped: {e}")
        additional_metadata_df = None
        _additional_metadata_mtime = _mtime_or_zero(ADDITIONAL_METADATA_FILE)


def _reload_samples_dashboard():
    """Load samples_dashboard.parquet into memory (small file)."""
    global samples_dashboard_df, _samples_dashboard_mtime
    try:
        samples_dashboard_df = pd.read_parquet(SAMPLES_DASHBOARD_FILE)
        _samples_dashboard_mtime = _mtime_or_zero(SAMPLES_DASHBOARD_FILE)
        print(
            f"✅ Loaded samples dashboard ({len(samples_dashboard_df)} rows) "
            f"from {SAMPLES_DASHBOARD_FILE}"
        )
    except Exception as e:
        print(f"⚠️ Samples dashboard load skipped: {e}")
        samples_dashboard_df = None
        _samples_dashboard_mtime = _mtime_or_zero(SAMPLES_DASHBOARD_FILE)


# Initial load + DB init
_load_data()
init_db()


@app.before_request
def _auto_sync():
    """Check parquet mtime; non-blocking reload if changed.

    Uses non-blocking lock acquisition so that concurrent requests are
    never stalled behind a slow reload (image downloads can take minutes).
    Requests that arrive while a reload is in progress simply proceed
    with the current (slightly stale but valid) data.
    """
    m_main = _mtime_or_zero(DATA_FILE)
    m_dash = _mtime_or_zero(SAMPLES_DASHBOARD_FILE)
    m_am = _mtime_or_zero(ADDITIONAL_METADATA_FILE)
    if (
        m_main == _parquet_mtime
        and m_dash == _samples_dashboard_mtime
        and m_am == _additional_metadata_mtime
    ):
        return

    # Non-blocking: if another thread is already reloading, serve current data
    acquired = _data_lock.acquire(blocking=False)
    if not acquired:
        return

    try:
        m_main = _mtime_or_zero(DATA_FILE)
        m_dash = _mtime_or_zero(SAMPLES_DASHBOARD_FILE)
        m_am = _mtime_or_zero(ADDITIONAL_METADATA_FILE)
        if (
            m_main == _parquet_mtime
            and m_dash == _samples_dashboard_mtime
            and m_am == _additional_metadata_mtime
        ):
            return
        if m_main != _parquet_mtime:
            print("📦 Parquet file changed on disk — reloading …")
            _load_data()
        else:
            if m_dash != _samples_dashboard_mtime:
                print("📦 Samples dashboard parquet changed — reloading …")
                _reload_samples_dashboard()
            if m_am != _additional_metadata_mtime:
                print("📦 Additional metadata parquet changed — reloading …")
                _reload_additional_metadata()
                _refresh_sample_numbers_union()
    finally:
        _data_lock.release()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def get_team() -> str | None:
    return session.get("team")


def is_logged_in() -> bool:
    return session.get("team") is not None


def is_admin() -> bool:
    return bool(session.get("is_admin"))


# ---------------------------------------------------------------------------
# Routes: Auth
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        account = TEAM_ACCOUNTS.get(username)
        if account and account["password"] == password:
            session["username"] = username
            session["team"] = account["team"]
            session["is_admin"] = bool(account.get("is_admin"))
            if session["is_admin"]:
                return redirect(url_for("admin_view"))
            return redirect(url_for("index"))
        error = "Invalid username or password"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Routes: Main page
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    if not is_logged_in():
        return redirect(url_for("login"))
    if is_admin():
        sample_q = request.args.get("sample")
        if sample_q:
            return redirect(url_for("admin_view", sample=sample_q))
        return redirect(url_for("admin_view"))
    return render_template(
        "index.html",
        team=session["team"],
        username=session["username"],
        sample_numbers=SAMPLE_NUMBERS,
        supercat_order=SUPERCAT_ORDER,
        supercat_labels=SUPERCAT_LABELS,
        supercat_groups=SUPERCAT_GROUPS,
        fine_standard=list(FINE_PROTOCOL_STANDARD.keys()),
        fine_ambiguous=list(FINE_PROTOCOL_AMBIGUOUS.keys()),
        all_categories=ALL_CATEGORIES,
    )


@app.route("/admin")
def admin_view():
    if not is_logged_in():
        return redirect(url_for("login"))
    if not is_admin():
        return redirect(url_for("index"))
    return render_template(
        "admin_view.html",
        team=session["team"],
        username=session["username"],
        sample_numbers=SAMPLE_NUMBERS,
        supercat_order=SUPERCAT_ORDER,
        supercat_labels=SUPERCAT_LABELS,
        supercat_groups=SUPERCAT_GROUPS,
        fine_ambiguous=list(FINE_PROTOCOL_AMBIGUOUS.keys()),
        all_categories=ALL_CATEGORIES,
        teams=TEAMS,
    )


# ---------------------------------------------------------------------------
# API: Sample Summary (includes this team's reviews)
# ---------------------------------------------------------------------------
@app.route("/api/sample/<int:sample_id>/summary")
def sample_summary(sample_id):
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401

    team = get_team()

    try:
        if df is None or df.empty:
            sample_df = pd.DataFrame()
        else:
            sample_df = df[df["sample_number"] == sample_id]

        am_for_sample = pd.DataFrame()
        if additional_metadata_df is not None and not additional_metadata_df.empty:
            am_for_sample = additional_metadata_df[
                additional_metadata_df["sample_number"] == sample_id
            ]

        if sample_df.empty and am_for_sample.empty:
            return jsonify({"error": "Sample not found"}), 404

        # --- Mixed form ---
        mixed_rows = sample_df[sample_df["form_type"] == MIXED_FORM]
        mixed_info = None
        mixed_images = []
        mixed_datapoint_id = None
        total_weight = None

        if not mixed_rows.empty:
            r = mixed_rows.iloc[0]
            raw_dp_id = r.get("datapoint_id")
            mixed_datapoint_id = str(raw_dp_id) if pd.notna(raw_dp_id) else None
            total_weight = sanitize(r.get("sample_weight"))
            mixed_info = {
                "user": sanitize(r.get("user")),
                "created_at": sanitize(r.get("created_at")),
                "updated_at": sanitize(r.get("updated_at")),
                "sample_weight": total_weight,
                "moisture": [
                    sanitize(r.get("sample_moisture1")),
                    sanitize(r.get("sample_moisture2")),
                    sanitize(r.get("sample_moisture3")),
                ],
                "background": sanitize(r.get("background_name")),
                "variety": sanitize(r.get("soybean_variety")),
            }
            for _, mrow in mixed_rows.iterrows():
                ref = sanitize(mrow.get("reference_object"))
                mixed_image_path = sanitize(mrow.get("local_image_path"))
                mixed_images.append({
                    "ref": ref,
                    "has_image": mixed_image_path is not None,
                    "image_path": mixed_image_path,
                    "sample_category": sanitize(mrow.get("sample_category")),
                })

        ref_order = {r: i for i, r in enumerate(MIXED_EXPECTED_REFS)}
        mixed_images.sort(key=lambda x: ref_order.get(x["ref"], 99))

        # --- Category forms ---
        cat_rows = sample_df[sample_df["form_type"].isin([CATEGORY_FORM, AMBIGUOUS_FORM])]
        collected = {}
        for _, crow in cat_rows.iterrows():
            cat_name = sanitize(crow.get("sample_category"))
            cat_weight = sanitize(crow.get(f"{cat_name}_weight"))
            if cat_name is None or cat_weight is None:
                continue
            cat_weight = float(cat_weight)

            raw_dp = crow.get("datapoint_id")
            entry = {
                "name": cat_name,
                "datapoint_id": str(raw_dp) if pd.notna(raw_dp) else None,
                "form_type": crow.get("form_type"),
                "supercategory": ALL_CATEGORIES.get(cat_name, "?"),
                "is_ambiguous": crow.get("form_type") == AMBIGUOUS_FORM,
                "exists": True,
                "weight": cat_weight,
                "has_image": pd.notna(crow.get("local_image_path")),
                "image_path": sanitize(crow.get("local_image_path")),
                "ref_object": sanitize(crow.get("reference_object")),
                "user": sanitize(crow.get("user")),
                "created_at": sanitize(crow.get("created_at")),
                "updated_at": sanitize(crow.get("updated_at")),
            }
            if crow.get("form_type") == AMBIGUOUS_FORM:
                entry.update({
                    "dg_weight": sanitize(crow.get("category_DG_weight")),
                    "gg_weight": sanitize(crow.get("category_GG_weight")),
                    "grg_weight": sanitize(crow.get("category_GrG_weight")),
                })
            collected[cat_name] = entry

        # Build full category list (collected + missing)
        all_cat_list = []
        sum_weights = 0.0
        for cat_name, supercat in ALL_CATEGORIES.items():
            if cat_name in collected:
                e = collected[cat_name]
                sum_weights += e["weight"]
                all_cat_list.append(e)
            else:
                all_cat_list.append({
                    "name": cat_name,
                    "datapoint_id": None,
                    "supercategory": supercat,
                    "is_ambiguous": cat_name in FINE_PROTOCOL_AMBIGUOUS,
                    "exists": False,
                    "weight": None,
                    "has_image": False,
                    "image_path": None,
                    "ref_object": None,
                    "user": None,
                    "created_at": None,
                })

        # --- Diagnostics ---
        total_w = float(total_weight) if total_weight is not None else 0.0
        diagnostics = {
            "mixed_exists": mixed_info is not None,
            "mixed_images_found": sum(1 for i in mixed_images if i["has_image"]),
            "mixed_images_expected": len(MIXED_EXPECTED_REFS),
            "missing_mixed_refs": [img["ref"] for img in mixed_images if not img["has_image"]],
            "total_weight": total_weight,
            "sum_category_weights": round(sum_weights, 3),
            "weight_diff": round(total_w - sum_weights, 3),
            "weight_diff_pct": round(
                ((total_w - sum_weights) / total_w * 100) if total_w else 0, 2
            ),
            "categories_collected": sum(1 for c in all_cat_list if c["exists"]),
            "categories_expected": len(ALL_CATEGORIES),
        }

        # --- Additional metadata forms (from pipeline export; not in main parquet) ---
        additional_metadata_out = []
        if not am_for_sample.empty:
            for _, am_row in am_for_sample.iterrows():
                raw_dp = am_row.get("datapoint_id")
                dp = str(raw_dp) if pd.notna(raw_dp) else None
                raw_json = am_row.get("data_json")
                try:
                    payload = json.loads(raw_json) if raw_json and pd.notna(raw_json) else {}
                except (json.JSONDecodeError, TypeError):
                    payload = {}
                additional_metadata_out.append({
                    "datapoint_id": dp,
                    "form_name": sanitize(am_row.get("form_name")),
                    "user": sanitize(am_row.get("user")),
                    "created_at": sanitize(am_row.get("created_at")),
                    "updated_at": sanitize(am_row.get("updated_at")),
                    "fields": _flatten_form_payload(payload),
                    "empty_state": False,
                })

        if not additional_metadata_out and not sample_df.empty:
            additional_metadata_out.append({
                "datapoint_id": _additional_metadata_absent_datapoint_id(sample_id),
                "form_name": DEFAULT_ADDITIONAL_METADATA_FORM_LABEL,
                "user": None,
                "created_at": None,
                "updated_at": None,
                "fields": [],
                "empty_state": True,
            })

        # --- Reviews (one team) or all teams for admin unified view ---
        if is_admin():
            reviews_out: dict = {}
            reviews_by_team = get_reviews_all_teams_for_sample(sample_id)
        else:
            reviews_out = get_reviews_for_sample(team, sample_id)
            reviews_by_team = None

        return jsonify({
            "sample_id": sample_id,
            "mixed": {
                "info": mixed_info,
                "images": mixed_images,
                "datapoint_id": mixed_datapoint_id,
            },
            "categories": all_cat_list,
            "diagnostics": diagnostics,
            "additional_metadata": additional_metadata_out,
            "reviews": reviews_out,
            "reviews_by_team": reviews_by_team,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# API: Reviews
# ---------------------------------------------------------------------------
@app.route("/api/review", methods=["POST"])
def api_save_review():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401
    if is_admin():
        return jsonify({"error": "Admin view is read-only"}), 403

    team = get_team()
    data = request.get_json()

    required = ["sample_number", "datapoint_id", "form_type", "verdict"]
    if not all(data.get(k) for k in required):
        return jsonify({"error": "Missing required fields"}), 400

    if data["verdict"] not in ("accept", "flag"):
        return jsonify({"error": "Invalid verdict"}), 400

    result = save_review(
        team=team,
        sample_number=data["sample_number"],
        datapoint_id=data["datapoint_id"],
        form_type=data["form_type"],
        sample_category=data.get("sample_category"),
        verdict=data["verdict"],
        issue_types=data.get("issue_types"),
        remark=data.get("remark"),
        reviewer_username=session.get("username"),
    )
    return jsonify(result)


@app.route("/api/review", methods=["DELETE"])
def api_delete_review():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401
    if is_admin():
        return jsonify({"error": "Admin view is read-only"}), 403

    team = get_team()
    data = request.get_json()
    dp_id = data.get("datapoint_id")
    if not dp_id:
        return jsonify({"error": "datapoint_id required"}), 400

    deleted = delete_review(team=team, datapoint_id=dp_id)
    if deleted:
        return jsonify({"ok": True})
    return jsonify({"ok": False, "message": "No review found"}), 404


# ---------------------------------------------------------------------------
# API: Issue Types
# ---------------------------------------------------------------------------
@app.route("/api/issue-types")
def api_get_issue_types():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(get_issue_types())


@app.route("/api/issue-types", methods=["POST"])
def api_add_issue_type():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401
    if is_admin():
        return jsonify({"error": "Admin view is read-only"}), 403

    data = request.get_json()
    label = (data.get("label") or "").strip()
    if not label:
        return jsonify({"error": "Label required"}), 400

    added = add_issue_type(label, session["team"])
    if added:
        return jsonify({"ok": True, "label": label})
    return jsonify({"ok": False, "message": "Already exists"}), 409


# ---------------------------------------------------------------------------
# Dashboard analytics (sample-level + collector-level tables)
# ---------------------------------------------------------------------------
def _sample_totals_from_main_df() -> dict[int, int]:
    """Unique datapoint_ids per sample from the image-review parquet."""
    if df is None or df.empty:
        return {}
    out: dict[int, int] = {}
    for sn, sdf in df.groupby("sample_number"):
        out[int(sn)] = int(sdf["datapoint_id"].nunique())
    return out


def _analytics_samples_fallback() -> list[dict]:
    """Approximate per-sample stats from latest.parquet only (no raw duplicates / metadata)."""
    if df is None or df.empty:
        return []
    work = df.copy()
    work["_created"] = pd.to_datetime(work["created_at"], utc=True, errors="coerce")
    work["_updated"] = pd.to_datetime(work["updated_at"], utc=True, errors="coerce")
    rows: list[dict] = []
    for sn, g in work.groupby("sample_number"):
        sn = int(sn)
        mixed = g[g["form_type"] == MIXED_FORM].sort_values("_created", ascending=True)
        if not mixed.empty:
            collected_by = str(mixed.iloc[0]["user"])
        else:
            collected_by = str(g.sort_values("_created", ascending=True).iloc[0]["user"])
        n_mixed = int(g[g["form_type"] == MIXED_FORM]["datapoint_id"].nunique())
        cat = g[g["form_type"].isin([CATEGORY_FORM, AMBIGUOUS_FORM])]
        n_sub = int(len(cat))
        n_sub_u = (
            int(cat.groupby(["form_type", "sample_category"]).ngroups) if n_sub else 0
        )
        fc = g["_created"].min()
        lu = g["_updated"].max()
        delta_s = None
        if pd.notna(fc) and pd.notna(lu):
            delta_s = float((lu - fc).total_seconds())
        rows.append(
            {
                "sample_number": sn,
                "collected_by": collected_by,
                "n_subcategory_forms": n_sub,
                "n_subcategory_forms_unique": n_sub_u,
                "n_mixed_forms": n_mixed,
                "n_additional_metadata_forms": 0,
                "first_form_created_at": fc,
                "last_form_updated_at": lu,
                "collection_time_seconds": delta_s,
            }
        )
    rows.sort(key=lambda r: r["sample_number"])
    return rows


def _json_safe_analytics_value(val):
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    # numpy scalar int/float
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            pass
    return val


def _canonical_team(team_raw: str) -> str | None:
    """Map DB team string onto TEAMS entry (case-insensitive, trimmed)."""
    t = str(team_raw).strip()
    for canonical in TEAMS:
        if t.lower() == canonical.lower():
            return canonical
    return None


def _merge_reviews_into_sample_rows(
    sample_rows: list[dict],
    sample_totals: dict[int, int],
) -> list[dict]:
    review_counts = get_review_counts_by_sample_and_team()
    idx: dict[tuple[int, str], dict] = {}
    for r in review_counts:
        sn_key = int(r["sample_number"])
        tr = str(r["team"]).strip()
        canon = _canonical_team(tr)
        key_team = canon if canon is not None else tr
        idx[(sn_key, key_team)] = dict(r)
    merged: list[dict] = []
    for row in sample_rows:
        sn = int(row["sample_number"])
        total = int(sample_totals.get(sn, 0))
        total_flags = 0
        out = dict(row)
        for t in TEAMS:
            rc = idx.get((sn, t), {})
            nf = int(rc.get("n_flagged", 0) or 0)
            na = int(rc.get("n_accepted", 0) or 0)
            nr = int(rc.get("n_reviewed", 0) or 0)
            total_flags += nf
            out[f"reviewed_{t}"] = nr
            out[f"accepted_{t}"] = na
            out[f"flagged_{t}"] = nf
        out["total_reviewable_forms"] = total
        out["flags_all_teams"] = total_flags
        merged.append(out)
    return merged


def _build_collectors_rows(sample_rows: list[dict]) -> list[dict]:
    by_user: dict[str, dict] = defaultdict(
        lambda: {"samples": set(), "times": [], "flagged_samples": set()}
    )
    for r in sample_rows:
        u = r.get("collected_by") or "Unknown"
        by_user[u]["samples"].add(int(r["sample_number"]))
        t = r.get("collection_time_seconds")
        if t is not None and not (isinstance(t, float) and math.isnan(t)):
            by_user[u]["times"].append(float(t))
        if int(r.get("flags_all_teams") or 0) > 0:
            by_user[u]["flagged_samples"].add(int(r["sample_number"]))
    out: list[dict] = []
    for user in sorted(by_user.keys(), key=str.lower):
        d = by_user[user]
        times = d["times"]
        med = float(pd.Series(times).median()) if times else None
        out.append(
            {
                "collector": user,
                "n_samples": len(d["samples"]),
                "n_samples_with_any_flag": len(d["flagged_samples"]),
                "median_collection_time_seconds": med,
            }
        )
    return out


_ISO_DT_PREFIX = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})",
)


def _truncate_display_datetime(val):
    """API display: YYYY-MM-DDTHH:MM:SS only (no subseconds / timezone tail)."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        ts = pd.Timestamp(val)
        if not pd.isna(ts):
            return ts.strftime("%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError, OverflowError):
        pass
    s = _json_safe_analytics_value(val)
    if not isinstance(s, str):
        return s
    s = s.strip()
    if len(s) >= 11 and s[10] == " ":
        s = s[:10] + "T" + s[11:]
    m = _ISO_DT_PREFIX.match(s)
    if m:
        return m.group(1).replace(" ", "T")
    if len(s) >= 19:
        return s[:19].replace(" ", "T", 1)
    return s


def _serialize_analytics_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k in ("first_form_created_at", "last_form_updated_at"):
            out[k] = _truncate_display_datetime(v)
        else:
            out[k] = _json_safe_analytics_value(v)
    return out


def build_dashboard_analytics_payload() -> dict:
    sample_totals = _sample_totals_from_main_df()
    if samples_dashboard_df is not None and not samples_dashboard_df.empty:
        base_rows = samples_dashboard_df.to_dict("records")
        source = "parquet"
    else:
        base_rows = _analytics_samples_fallback()
        source = "fallback"
    merged = _merge_reviews_into_sample_rows(base_rows, sample_totals)
    samples_out = [_serialize_analytics_row(r) for r in merged]
    collectors_out = [_serialize_analytics_row(r) for r in _build_collectors_rows(merged)]
    return {
        "samples": samples_out,
        "collectors": collectors_out,
        "source": source,
        "teams": TEAMS,
    }


# ---------------------------------------------------------------------------
# API: Dashboard stats
# ---------------------------------------------------------------------------
@app.route("/api/dashboard")
def api_dashboard():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401

    # Total reviewable forms per sample = unique datapoint_ids
    sample_totals = {}
    for sn in SAMPLE_NUMBERS:
        sdf = df[df["sample_number"] == sn]
        sample_totals[sn] = sdf["datapoint_id"].nunique()

    all_reviews = get_all_reviews()

    # Aggregate per team per sample
    team_sample_stats: dict[str, dict] = {t: {} for t in TEAMS}
    for rev in all_reviews:
        t_raw = str(rev["team"]).strip()
        t = _canonical_team(t_raw)
        if t is None:
            continue
        sn = rev["sample_number"]
        if sn not in team_sample_stats[t]:
            team_sample_stats[t][sn] = {"reviewed": 0, "accepted": 0, "flagged": 0}
        team_sample_stats[t][sn]["reviewed"] += 1
        if rev["verdict"] == "accept":
            team_sample_stats[t][sn]["accepted"] += 1
        else:
            team_sample_stats[t][sn]["flagged"] += 1

    grand_total = sum(sample_totals.values())

    teams_summary = []
    for t in TEAMS:
        total_reviewed = sum(
            s.get("reviewed", 0) for s in team_sample_stats[t].values()
        )
        total_accepted = sum(
            s.get("accepted", 0) for s in team_sample_stats[t].values()
        )
        total_flagged = sum(
            s.get("flagged", 0) for s in team_sample_stats[t].values()
        )
        teams_summary.append({
            "team": t,
            "reviewed": total_reviewed,
            "total": grand_total,
            "accepted": total_accepted,
            "flagged": total_flagged,
            "pct": round(total_reviewed / grand_total * 100, 1)
                   if grand_total > 0 else 0,
        })

    samples = []
    for sn in SAMPLE_NUMBERS:
        total = sample_totals.get(sn, 0)
        by_team = {}
        for t in TEAMS:
            by_team[t] = team_sample_stats[t].get(
                sn, {"reviewed": 0, "accepted": 0, "flagged": 0}
            )
        samples.append({
            "sample_number": sn,
            "total_forms": total,
            "by_team": by_team,
        })

    return jsonify({
        "teams": teams_summary,
        "samples": samples,
        "grand_total": grand_total,
    })


@app.route("/api/dashboard/analytics")
def api_dashboard_analytics():
    """Sortable/filterable sample + collector tables (merged with reviews.db)."""
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(build_dashboard_analytics_payload())


# ---------------------------------------------------------------------------
# Routes: Dashboard page
# ---------------------------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template(
        "dashboard.html",
        team=session["team"],
        username=session["username"],
    )


# ---------------------------------------------------------------------------
# Routes: Image download
# ---------------------------------------------------------------------------
@app.route("/download/<path:filename>")
def download_image(filename):
    if not is_logged_in():
        return redirect(url_for("login"))
    return send_from_directory(str(STATIC_IMG_DIR), filename, as_attachment=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Teams: {', '.join(TEAMS)}")
    print(f"Accounts: {', '.join(TEAM_ACCOUNTS.keys())}")
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=7862, debug=debug, use_reloader=debug)
