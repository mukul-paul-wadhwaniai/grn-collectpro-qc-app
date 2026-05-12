import os
import secrets
import threading
from uuid import uuid4
from collections import defaultdict

import pandas as pd
from pathlib import Path
from flask import (
    Flask, render_template, jsonify, request,
    send_from_directory, session, redirect, url_for,
)
from google.cloud import storage

from db import (
    init_db, save_review, delete_review, get_reviews_for_sample,
    get_all_reviews, get_issue_types, add_issue_type,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_FILE = "/data/temp_dir/latest.parquet"
STATIC_IMG_DIR = Path("static/images")

# Team accounts: username → { password, team }
TEAM_ACCOUNTS = {
    "ml":         {"password": "qM565~v-Tw\K", "team": "ML"},
    "programmes": {"password": "qM565~v-Tw\K", "team": "Programmes"},
    "product":    {"password": "qM565~v-Tw\K", "team": "Product"},
}
TEAMS = ["ML", "Programmes", "Product"]

try:
    GCS_CLIENT = storage.Client()
except Exception as e:
    print(f"ERROR: Failed to initialize GCS client (check credentials): {e}")
    exit(1)

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

STATIC_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Form Types
MIXED_FORM = "MixedSample_FineProtocol"
CATEGORY_FORM = "CategorySample_FineProtocol"
AMBIGUOUS_FORM = "CategorySample_Ambiguous_FineProtocol"

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
# GCP Utility
# ---------------------------------------------------------------------------
def download_image_from_gcs_bucket(gcp_url: str, temp_dir: Path = Path("tmp"),
                                   imagename: str | None = None):
    """Download a blob from GCP using a gs:// URL."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        parts = gcp_url.split("/")
        if len(parts) < 4 or parts[0] != "gs:":
            print(f"ERROR: Invalid GCP URL format: {gcp_url}")
            return None
        bucket_name = parts[2]
        blob_name = "/".join(parts[3:])
        bucket = GCS_CLIENT.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            print(f"ERROR: Blob {blob_name} does not exist in bucket {bucket_name}.")
            return None
        temp_path = temp_dir / (imagename or f"{uuid4()}.jpg")
        blob.download_to_filename(temp_path)
        return str(temp_path.resolve())
    except Exception as e:
        print(f"ERROR: Error processing {gcp_url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Pre-fetch images
# ---------------------------------------------------------------------------
def pre_download_images(dataframe):
    print("Checking and pre-fetching images from GCP …")
    local_paths = []
    for _, row in dataframe.iterrows():
        gcp_url = row.get("gcp_url")
        if not gcp_url or pd.isna(gcp_url):
            local_paths.append(None)
            continue
        filename = gcp_url.split("/")[-1]
        local_filepath = STATIC_IMG_DIR / filename
        if not local_filepath.exists():
            download_image_from_gcs_bucket(
                gcp_url=gcp_url,
                temp_dir=Path(STATIC_IMG_DIR),
                imagename=filename,
            )
        if local_filepath.exists():
            local_paths.append(f"/static/images/{filename}")
        else:
            local_paths.append(None)
    dataframe["local_image_path"] = local_paths
    print("GCP image sync complete.")
    return dataframe


# ---------------------------------------------------------------------------
# NaN / None helper
# ---------------------------------------------------------------------------
def sanitize(val):
    """Convert NaN / None to Python None for JSON safety."""
    if pd.isna(val):
        return None
    return val


# ---------------------------------------------------------------------------
# Auto-sync: reload parquet when the file changes on disk
# ---------------------------------------------------------------------------
_data_lock = threading.Lock()
_parquet_mtime: float = 0.0
df: pd.DataFrame | None = None
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
        new_samples = sorted(
            new_df["sample_number"].dropna().unique().astype(int).tolist()
        )
        new_mtime = os.path.getmtime(DATA_FILE)
        # Atomic swap — only reached if everything above succeeded
        df = new_df
        SAMPLE_NUMBERS = new_samples
        _parquet_mtime = new_mtime
        print(f"✅ Loaded {len(df)} rows, {len(SAMPLE_NUMBERS)} samples")
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
    try:
        current_mtime = os.path.getmtime(DATA_FILE)
    except OSError:
        return
    if current_mtime == _parquet_mtime:
        return

    # Non-blocking: if another thread is already reloading, serve current data
    acquired = _data_lock.acquire(blocking=False)
    if not acquired:
        return

    try:
        # Double-check after acquiring lock (another thread may have just finished)
        try:
            if os.path.getmtime(DATA_FILE) == _parquet_mtime:
                return
        except OSError:
            return
        print("📦 Parquet file changed on disk — reloading …")
        _load_data()
    finally:
        _data_lock.release()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def get_team() -> str | None:
    return session.get("team")


def is_logged_in() -> bool:
    return session.get("team") is not None


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


# ---------------------------------------------------------------------------
# API: Sample Summary (includes this team's reviews)
# ---------------------------------------------------------------------------
@app.route("/api/sample/<int:sample_id>/summary")
def sample_summary(sample_id):
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401

    team = get_team()

    try:
        sample_df = df[df["sample_number"] == sample_id]
        if sample_df.empty:
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

        # --- Reviews for this team ---
        reviews = get_reviews_for_sample(team, sample_id)

        return jsonify({
            "sample_id": sample_id,
            "mixed": {
                "info": mixed_info,
                "images": mixed_images,
                "datapoint_id": mixed_datapoint_id,
            },
            "categories": all_cat_list,
            "diagnostics": diagnostics,
            "reviews": reviews,
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
    )
    return jsonify(result)


@app.route("/api/review", methods=["DELETE"])
def api_delete_review():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401

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

    data = request.get_json()
    label = (data.get("label") or "").strip()
    if not label:
        return jsonify({"error": "Label required"}), 400

    added = add_issue_type(label, session["team"])
    if added:
        return jsonify({"ok": True, "label": label})
    return jsonify({"ok": False, "message": "Already exists"}), 409


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
        t = rev["team"]
        sn = rev["sample_number"]
        if t not in team_sample_stats:
            team_sample_stats[t] = {}
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
    app.run(host="0.0.0.0", port=7865, debug=False)
