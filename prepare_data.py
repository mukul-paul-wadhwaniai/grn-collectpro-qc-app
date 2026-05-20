import json
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from util import (
    fetch_latest_data,
    setup_df,
    logger,
)

DEBUG_SAMPLE_NUMBERS = [0, 1, 2]

S3_BUCKET_NAME = 'agri-grn-prod-dip-bucket'
PROJECT_NAME = "GRN WIAI Round 3 Collection"
SOURCE = "wiai_round_3_collection"

# Form Constants
MIXED_FORM = "MixedSample_FineProtocol"
CATEGORY_FORM = "CategorySample_FineProtocol"
AMBIGUOUS_FORM = "CategorySample_Ambiguous_FineProtocol"
FORM_NAMES = [MIXED_FORM, CATEGORY_FORM, AMBIGUOUS_FORM]

# Optional extra form types (e.g. Additional Metadata) — comma-separated
# exact names as in projectapp_dataset.name / projectapp_datapoint join.
# EXTRA_FORM_NAMES = [
#     x.strip()
#     for x in os.environ.get("GRN_EXTRA_FORM_NAMES", "").split(",")
#     if x.strip()
# ]
EXTRA_FORM_NAMES = [ "Additional metadata fine protocol", "Additional Metadata Fine Protocol" ]
ALL_FORM_NAMES = list(dict.fromkeys(FORM_NAMES + EXTRA_FORM_NAMES))

# list of users for this season's collection
VALID_USERS = ["kailas", "Vipin", "Pradip"]

# Standardize Reference Objects
REF_NAME_MAPPING = {
    'A4 sheet': 'a4_sheet',
    'Rs 50 note': '50_note',
    'Rs 100 note': '100_note',
    'empty_image': 'absent',
    'sample_with_a4_sheet': 'a4_sheet',
    'sample_with_rs_50_note': '50_note',
    'sample_with_rs_100_note': '100_note'
}


def extract_sample_number(data) -> int | None:
    """
    Resolve sample_number from form JSON.

    Mixed/category forms use data['sample_metadata']['sample_number'].
    Additional-metadata forms store sample_number at the top level of data.
    """
    if not isinstance(data, dict):
        return None
    sample_meta = data.get("sample_metadata")
    if isinstance(sample_meta, dict):
        sn = sample_meta.get("sample_number")
        if sn is not None and sn != "":
            try:
                return int(sn)
            except (TypeError, ValueError):
                pass
    sn = data.get("sample_number")
    if sn is not None and sn != "":
        try:
            return int(sn)
        except (TypeError, ValueError):
            pass
    return None


# Output directory for timestamped parquet archives
OUTPUT_DIR = Path("/data/temp_dir") # NOTE: this is a temporary directory for testing
PROCESSED_SYMLINK = OUTPUT_DIR / "latest.parquet"
ADDITIONAL_METADATA_SYMLINK = OUTPUT_DIR / "additional_metadata.parquet"


def validate_and_dedup(df: pd.DataFrame, is_processed=False):
    """
    Validate sample integrity and deduplicate:
    - If multiple mixed forms exist for a sample, keep the latest (by created_at).
    - If multiple category forms exist for the same (form_name/form_type, sample_category),
      keep the latest (by created_at).
    Always logs warnings about duplicates found.
    Returns cleaned_df.
    """
    drop_ids = set()  # datapoint_ids to remove

    # Handle column name differences between raw and processed dfs
    form_col = 'form_type' if is_processed else 'form_name'

    for sample_num, group_df in df.groupby('sample_number'):
        # --- Dedup mixed forms ---
        mixed_df = group_df[group_df[form_col] == MIXED_FORM]

        # in processed data, 1 mixed form = 4 rows. So, we need to count unique datapoint_ids for mixed forms.
        unique_mixed_ids = mixed_df['datapoint_id'].nunique()
        if unique_mixed_ids == 0:
            logger.warning(f"Sample {int(sample_num)}: 0 mixed forms (expected 1)")
        elif unique_mixed_ids > 1:
            # Keep the one with the latest created_at
            unique_mixed_df = mixed_df.drop_duplicates(subset=['datapoint_id'])
            sorted_mixed = unique_mixed_df.sort_values('created_at', ascending=False)
            keep_id = sorted_mixed.iloc[0]['datapoint_id']
            stale_ids = sorted_mixed.iloc[1:]['datapoint_id'].tolist()
            drop_ids.update(stale_ids)
            logger.warning(
                f"Sample {int(sample_num)}: {unique_mixed_ids} mixed forms — "
                f"keeping {keep_id}, dropping {stale_ids}"
            )

        # --- Dedup category forms ---
        cat_df = group_df[group_df[form_col].isin([CATEGORY_FORM, AMBIGUOUS_FORM])]
        if not cat_df.empty:
            cat_df = cat_df.copy()

            # Extract category depending on df format
            if is_processed:
                cat_df['_cat_key'] = cat_df.apply(
                    lambda row: (row[form_col], row['sample_category']), axis=1
                )
            else:
                cat_df['_cat_key'] = cat_df.apply(
                    lambda row: (
                        row[form_col],
                        row['data']['sample_metadata']['sample_category']
                    ),
                    axis=1,
                )
            
            # Group by the category key and check for duplicate datapoint_ids
            dup_mask = cat_df['_cat_key'].duplicated(keep=False)
            if dup_mask.any():
                # For each duplicate group, keep latest
                for key, dup_group in cat_df[dup_mask].groupby('_cat_key'):
                    sorted_group = dup_group.sort_values('created_at', ascending=False)
                    keep_id = sorted_group.iloc[0]['datapoint_id']
                    stale_ids = sorted_group.iloc[1:]['datapoint_id'].tolist()
                    drop_ids.update(stale_ids)
                    logger.warning(
                        f"Sample {int(sample_num)}: duplicate category {key} — "
                        f"keeping {keep_id}, dropping {stale_ids}"
                    )

    # --- Apply drops ---
    if drop_ids:
        before = len(df)
        df = df[~df['datapoint_id'].isin(drop_ids)]
        logger.info(
            f"Deduplication: dropped {before - len(df)} stale rows "
            f"({len(drop_ids)} datapoint_ids). "
            f"Remaining: {df['sample_number'].nunique()} samples, {len(df)} rows"
        )
    else:
        logger.info("All samples passed integrity check — no duplicates found.")

    return df


def get_latest_parquet(output_dir: Path) -> Path | None:
    """Find the most recent parquet file in the output directory by name (YYYYMMDD_HHMMSS)."""
    parquets = sorted(
        p
        for p in output_dir.glob("*.parquet")
        if not p.is_symlink() and not p.name.startswith("samples_dashboard_")
    )
    return parquets[-1] if parquets else None


def _category_keys_vectorized(cat_df: pd.DataFrame) -> pd.Series:
    """(form_name, sample_category) per row for category / ambiguous forms."""
    def _cat(row):
        try:
            meta = row["data"].get("sample_metadata", {})
            return (row["form_name"], meta.get("sample_category"))
        except Exception:
            return (row["form_name"], None)

    return cat_df.apply(_cat, axis=1)


def build_samples_dashboard_summary(
    raw_df: pd.DataFrame,
    metadata_form_names: list[str],
) -> pd.DataFrame:
    """
    One row per sample_number from raw (pre-dedup) datapoints.

    Counts include all submitted forms so duplicates are visible.
    """
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "sample_number",
                "collected_by",
                "n_subcategory_forms",
                "n_subcategory_forms_unique",
                "n_mixed_forms",
                "n_additional_metadata_forms",
                "first_form_created_at",
                "last_form_updated_at",
                "collection_time_seconds",
            ]
        )

    work = raw_df.copy()
    work["created_at"] = pd.to_datetime(work["created_at"], utc=True, errors="coerce")
    work["updated_at"] = pd.to_datetime(work["updated_at"], utc=True, errors="coerce")

    meta_set = set(metadata_form_names)
    rows = []
    for sn, g in work.groupby("sample_number", sort=True):
        sn = int(sn)
        first_created = g["created_at"].min()
        last_updated = g["updated_at"].max()
        if pd.isna(first_created) or pd.isna(last_updated):
            delta_s = None
        else:
            delta_s = (last_updated - first_created).total_seconds()

        mixed_g = g[g["form_name"] == MIXED_FORM].sort_values("created_at", ascending=True)
        if not mixed_g.empty:
            collected_by = str(mixed_g.iloc[0]["username"])
        else:
            g_sorted = g.sort_values("created_at", ascending=True)
            collected_by = str(g_sorted.iloc[0]["username"])

        n_mixed = g[g["form_name"] == MIXED_FORM]["datapoint_id"].nunique()

        cat_g = g[g["form_name"].isin([CATEGORY_FORM, AMBIGUOUS_FORM])]
        n_sub = int(len(cat_g))
        if len(cat_g) == 0:
            n_sub_u = 0
        else:
            keys = _category_keys_vectorized(cat_g)
            n_sub_u = int(keys.nunique())

        if meta_set:
            n_meta = int(g[g["form_name"].isin(meta_set)]["datapoint_id"].nunique())
        else:
            n_meta = 0

        rows.append(
            {
                "sample_number": sn,
                "collected_by": collected_by,
                "n_subcategory_forms": n_sub,
                "n_subcategory_forms_unique": n_sub_u,
                "n_mixed_forms": int(n_mixed),
                "n_additional_metadata_forms": n_meta,
                "first_form_created_at": first_created,
                "last_form_updated_at": last_updated,
                "collection_time_seconds": float(delta_s) if delta_s is not None else None,
            }
        )

    out = pd.DataFrame(rows).sort_values("sample_number").reset_index(drop=True)
    return out


def write_samples_dashboard_parquet(df: pd.DataFrame, output_dir: Path) -> Path:
    """Write timestamped parquet and atomically update samples_dashboard symlink."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"samples_dashboard_{timestamp}.parquet"
    df.to_parquet(out_path, index=False)

    tmp_link = output_dir / f".samples_dash_{os.getpid()}.tmp"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    tmp_link.symlink_to(out_path)
    tmp_link.rename(output_dir / "samples_dashboard.parquet")
    logger.info(f"✅ Samples dashboard saved → {out_path.name} (symlink updated)")
    return out_path


def build_additional_metadata_export(
    raw_df: pd.DataFrame,
    metadata_form_names: list[str],
) -> pd.DataFrame:
    """
    Rows for Additional Metadata (and similar) forms — used by the review UI.
    Serialized `data` JSON per datapoint.
    """
    cols = [
        "sample_number",
        "datapoint_id",
        "form_name",
        "user",
        "created_at",
        "updated_at",
        "data_json",
    ]
    if raw_df.empty or not metadata_form_names:
        return pd.DataFrame(columns=cols)

    meta_set = set(metadata_form_names)
    m = raw_df[raw_df["form_name"].isin(meta_set)].copy()
    if m.empty:
        return pd.DataFrame(columns=cols)

    m["data_json"] = m["data"].apply(lambda d: json.dumps(d, default=str))
    out = m[
        ["sample_number", "datapoint_id", "form_name", "username", "created_at", "updated_at"]
    ].rename(columns={"username": "user"})
    out["data_json"] = m["data_json"].values
    out = out.sort_values(["sample_number", "created_at"]).reset_index(drop=True)
    return out


def write_additional_metadata_parquet(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Write timestamped parquet and atomically update additional_metadata symlink."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        empty_path = output_dir / "_additional_metadata_empty.parquet"
        df.to_parquet(empty_path, index=False)
        tmp_link = output_dir / f".additional_meta_{os.getpid()}.tmp"
        if tmp_link.is_symlink() or tmp_link.exists():
            tmp_link.unlink()
        tmp_link.symlink_to(empty_path)
        tmp_link.rename(ADDITIONAL_METADATA_SYMLINK)
        logger.info("✅ Additional metadata export (empty) symlink updated")
        return empty_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"additional_metadata_{timestamp}.parquet"
    df.to_parquet(out_path, index=False)

    tmp_link = output_dir / f".additional_meta_{os.getpid()}.tmp"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    tmp_link.symlink_to(out_path)
    tmp_link.rename(ADDITIONAL_METADATA_SYMLINK)
    logger.info(f"✅ Additional metadata saved → {out_path.name} (symlink updated)")
    return out_path


def process_sample_group(sample_number, sample_df, skip_datapoint_ids=None):
    """
    Processes all forms associated with a single sample_number.
    Extracts shared metadata from the Mixed form and applies it to Category forms.

    If skip_datapoint_ids is provided, only generates output rows for datapoints
    NOT in that set (but still reads the Mixed form for shared context).
    """
    skip_datapoint_ids = skip_datapoint_ids or set()
    processed_rows = []
    
    # 1. Isolate the Mixed Sample Form to extract shared metadata
    mixed_df = sample_df[sample_df['form_name'] == MIXED_FORM]
    if mixed_df.empty:
        logger.warning(f"No Mixed Sample Form found for sample {sample_number}. Skipping.")
        return []
        
    mixed_row = mixed_df.iloc[0]
    mixed_data = mixed_row['data']
    mixed_metadata = mixed_data['sample_metadata']
    
    # Shared metadata for this sample
    shared_context = {
        'sample_number': sample_number,
        'source': SOURCE,
        'background_name': mixed_metadata['background'], # mandatory field in the form
        'soybean_variety': mixed_metadata.get('variety'), # optional field in the form
    }

    # Extract moisture values
    moisture_values = {
        'sample_moisture1': mixed_metadata.get('moisture_value_1_'), # optional field in the form
        'sample_moisture2': mixed_metadata.get('moisture_value_2'), # optional field in the form
        'sample_moisture3': mixed_metadata.get('moisture_value_3'), # optional field in the form
    }

    # Need total_weight for category rows even if mixed form was already processed
    total_weight = float(mixed_metadata['weight_(in_grams)']) # mandatory field in the form

    # 2. Process Mixed Form Images - only if this mixed form is new
    if mixed_row['datapoint_id'] not in skip_datapoint_ids:
        image_section = mixed_data['image_capture_section']
        for img_key in ['empty_image', 'sample_with_a4_sheet', 'sample_with_rs_50_note', 'sample_with_rs_100_note']:
            processed_rows.append({
                'datapoint_id': mixed_row['datapoint_id'],
                **shared_context,
                **moisture_values,
                'created_at': mixed_row['created_at'],
                'updated_at': mixed_row['updated_at'],
                'user': mixed_row['username'],
                'form_type': MIXED_FORM,
                'sample_category': 'empty' if img_key == 'empty_image' else 'mixed',
                'reference_object': REF_NAME_MAPPING[img_key],
                'sample_weight': total_weight,
                's3_url': image_section[img_key][0] if len(image_section[img_key]) > 0 else None
            })

    # 3. Process Standard and Ambiguous Category Forms - only new ones
    category_df = sample_df[sample_df['form_name'].isin([CATEGORY_FORM, AMBIGUOUS_FORM])]
    
    for _, cat_row in category_df.iterrows():
        if cat_row['datapoint_id'] in skip_datapoint_ids:
            continue  # Already processed so skip
        
        cat_data = cat_row['data']
        cat_metadata = cat_data['sample_metadata']
        form_name = cat_row['form_name']
        
        # Get Reference Object
        ref_dict = cat_data['select_one_of_the_reference_objects_from_the_dropdown_(use_the_same_reference_object_for_all_the_categories_and_the_ambiguous_categories_in_a_sample)']
        raw_ref_object = ref_dict['reference_object']
        
        # Get Image
        cat_img_section = cat_data['image_capture_section']
        img_url = cat_img_section['sample_with_selected_reference_object']

        entry = {
            'datapoint_id': cat_row['datapoint_id'],
            **shared_context,
            'created_at': cat_row['created_at'],
            'updated_at': cat_row['updated_at'],
            'user': cat_row['username'],
            'form_type': form_name,
            # sample_category: obvious_good_grains, obvious_good_split_grains, purple_grains, soiled_grains, obvious_damaged_grains, obvious_green_grains, small_good_grains, small_split_grains, peels, mud_balls, wood, small_particles, bengal_gram, wheat, pigeon_pea, sorghum, maize, black_gram, green_gram, other
            'sample_category': cat_metadata['sample_category'],
            'sample_weight': total_weight,
            f'{cat_metadata["sample_category"]}_weight': float(cat_metadata['weight_(in_grams)']),
            'reference_object': REF_NAME_MAPPING[raw_ref_object],
            's3_url': img_url[0] if len(img_url) > 0 else None
        }

        # 4. Handle Ambiguous Specific Supercategory Weights
        if form_name == AMBIGUOUS_FORM:
            super_cat_section = cat_data.get('supercategory_mapping_section_(bite_all_grains_and_segregate_into_true_supercategory)', {})
            entry.update({
                'category_DG_weight': super_cat_section.get('weight_of_dg_supercategory_'),
                'category_GG_weight': super_cat_section.get('weight_of_gg_supercategory_'),
                'category_GrG_weight': super_cat_section.get('weight_of_grg_supercategory_')
            })

        processed_rows.append(entry)

    return processed_rows

def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # 1. Fetch fresh CSVs from DB
    fetch_latest_data()

    # 2. Build raw dataframe including optional extra forms (metadata, etc.)
    raw_full = setup_df(
        valid_users=VALID_USERS,
        form_names=ALL_FORM_NAMES,
        project_name=PROJECT_NAME,
    )

    # Extract sample_number for grouping (schema differs by form type)
    raw_full["sample_number"] = raw_full["data"].apply(extract_sample_number)

    assert not raw_full["datapoint_id"].isnull().any(), "found None datapoint_id"
    missing_sn = raw_full["sample_number"].isnull()
    if missing_sn.any():
        by_form = raw_full.loc[missing_sn, "form_name"].value_counts().to_dict()
        logger.warning(
            f"Dropping {int(missing_sn.sum())} datapoints with no sample_number: {by_form}"
        )
        raw_full = raw_full[~missing_sn].copy()

    # 2b. Unified per-sample dashboard table (pre-dedup counts, all form types)
    dashboard_df = build_samples_dashboard_summary(raw_full, EXTRA_FORM_NAMES)
    write_samples_dashboard_parquet(dashboard_df, OUTPUT_DIR)

    meta_export_df = build_additional_metadata_export(raw_full, EXTRA_FORM_NAMES)
    write_additional_metadata_parquet(meta_export_df, OUTPUT_DIR)

    # 2c. Pipeline subset: only image-bearing protocol forms
    raw_df = raw_full[raw_full["form_name"].isin(FORM_NAMES)].copy()

    raw_df = validate_and_dedup(raw_df, is_processed=False)

    # 3. Load existing parquet (if any) to find already-processed datapoint_ids
    latest_parquet = get_latest_parquet(OUTPUT_DIR)
    existing_df = None
    already_processed_ids = set()
    
    if latest_parquet:
        existing_df = pd.read_parquet(latest_parquet)
        already_processed_ids = set(existing_df["datapoint_id"].unique())

        if "s3_url" in existing_df.columns:
            incomplete_ids = set(
                existing_df[existing_df["s3_url"].isna()]["datapoint_id"].unique()
            )
            existing_df = existing_df[~existing_df["datapoint_id"].isin(incomplete_ids)]
            already_processed_ids -= incomplete_ids
        else:
            logger.warning(
                "Existing latest parquet has no 's3_url' column (legacy export) — "
                "not dropping rows with missing URLs."
            )

        logger.info(
            f"Loaded {latest_parquet.name}: "
            f"{len(already_processed_ids)} already-processed datapoints, "
        )

    # 4. Identify new datapoints
    new_mask = ~(raw_df['datapoint_id'].isin(already_processed_ids))
    n_new = new_mask.sum()

    if n_new == 0:
        logger.info("No new entries to process. Everything is up to date.")
        return

    # 5. Get sample_numbers that have at least one new datapoint
    samples_with_new_data = set(
        raw_df.loc[new_mask, 'sample_number'].unique()
    )
    logger.info(
        f"Found {n_new} new datapoints across {len(samples_with_new_data)} samples"
    )

    # 6. Process - load full sample groups (for context) but only output new rows
    #    We need the full sample group (including already-processed mixed form)
    #    so that new category rows can access shared metadata.
    samples_to_process = raw_df[raw_df['sample_number'].isin(samples_with_new_data)]
    
    new_rows = []
    for sample_num, group_df in tqdm(
        samples_to_process.groupby('sample_number'), desc='Processing new entries'
    ):
        sample_num = int(sample_num)
        rows = process_sample_group(
            sample_num, group_df, 
            skip_datapoint_ids=already_processed_ids
        )
        new_rows.extend(rows)

    if not new_rows:
        logger.info("No new rows generated (all new datapoints may lack Mixed forms).")
        return

    new_df = pd.DataFrame(new_rows)
    logger.info(f"Generated {len(new_df)} new rows")

    # Images stay on S3 (s3_url in parquet). The review app downloads them to VM disk on load.
    new_processed = new_df

    # 8. Merge with existing data
    if existing_df is not None:
        final_output_df = pd.concat([existing_df, new_processed], ignore_index=True)
        # dedup final df
        final_output_df = validate_and_dedup(final_output_df, is_processed=True)
    else:
        final_output_df = new_processed

    # 9. Save timestamped parquet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{timestamp}.parquet"
    final_output_df.to_parquet(out_path, index=False)

    # 10. Update symlink ATOMICALLY so the app never sees a missing file.
    #     Create a temporary symlink then rename (atomic on Linux).
    tmp_link = OUTPUT_DIR / f".latest_{os.getpid()}.tmp"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    tmp_link.symlink_to(out_path)
    tmp_link.rename(PROCESSED_SYMLINK)  # atomic on same filesystem

    logger.info(
        f"✅ Saved {out_path.name}: "
        f"{len(new_processed)} new + "
        f"{len(existing_df) if existing_df is not None else 0} existing = "
        f"{len(final_output_df)} total rows"
    )


if __name__ == '__main__':
    main()