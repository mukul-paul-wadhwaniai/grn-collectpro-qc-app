import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from util import (
    fetch_latest_data,
    setup_df,
    push_images_to_gcp,
    logger
)

DEBUG_SAMPLE_NUMBERS = [0, 1, 2]

S3_BUCKET_NAME = 'agri-grn-prod-dip-bucket'
GCS_BUCKET_NAME = "bucket-prod-grn-asso1-fga-vm-data"
PROJECT_NAME = "GRN WIAI Round 3 Collection"
GCS_ROOT = f"temp_dir"
SOURCE = "wiai_round_3_collection"

# Form Constants
MIXED_FORM = "MixedSample_FineProtocol"
CATEGORY_FORM = "CategorySample_FineProtocol"
AMBIGUOUS_FORM = "CategorySample_Ambiguous_FineProtocol"
FORM_NAMES = [MIXED_FORM, CATEGORY_FORM, AMBIGUOUS_FORM]

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


# Output directory for timestamped parquet archives
OUTPUT_DIR = Path("/data/temp_dir") # NOTE: this is a temporary directory for testing
PROCESSED_SYMLINK = OUTPUT_DIR / "latest.parquet"


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
    parquets = sorted(p for p in output_dir.glob("*.parquet") if not p.is_symlink())
    return parquets[-1] if parquets else None


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
    
    # 2. Build raw dataframe (all valid entries from DB)
    raw_df = setup_df(
        valid_users=VALID_USERS, 
        form_names=FORM_NAMES, 
        project_name=PROJECT_NAME
    )

    # Extract sample_number for grouping
    raw_df['sample_number'] = raw_df['data'].apply(lambda x: x.get('sample_metadata', {}).get('sample_number'))

    # assert none of the datapoint_ids, sample_numbers are None
    assert not raw_df['datapoint_id'].isnull().any(), "found None datapoint_id"
    assert not raw_df['sample_number'].isnull().any(), "found None sample_number"

    # TODO: Figure out how to do this once debugging stage is over
    # drop samples with sample_number in DEBUG_SAMPLE_NUMBERS
    # raw_df = raw_df[~raw_df['sample_number'].isin(DEBUG_SAMPLE_NUMBERS)]

    raw_df = validate_and_dedup(raw_df, is_processed=False)

    # 3. Load existing parquet (if any) to find already-processed datapoint_ids
    latest_parquet = get_latest_parquet(OUTPUT_DIR)
    existing_df = None
    already_processed_ids = set()
    
    if latest_parquet:
        existing_df = pd.read_parquet(latest_parquet)
        already_processed_ids = set(existing_df['datapoint_id'].unique())

        incomplete_ids = set(existing_df[existing_df['s3_url'].isna()]['datapoint_id'].unique())
        existing_df = existing_df[~existing_df['datapoint_id'].isin(incomplete_ids)]
        already_processed_ids -= incomplete_ids

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

    # 7. Push ONLY new images to GCP
    mixed_new = new_df[new_df["form_type"] == MIXED_FORM]
    cat_new = new_df[new_df["form_type"].isin([CATEGORY_FORM, AMBIGUOUS_FORM])]

    mixed_prefix = Path(f"{GCS_ROOT}/mixed_form")
    category_prefix = Path(f"{GCS_ROOT}/category_forms")

    if not mixed_new.empty:
        logger.info(f"Uploading {len(mixed_new)} new mixed form images to GCP...")
        mixed_new = push_images_to_gcp(mixed_new, mixed_prefix, GCS_BUCKET_NAME, S3_BUCKET_NAME)

    if not cat_new.empty:
        logger.info(f"Uploading {len(cat_new)} new category form images to GCP...")
        cat_new = push_images_to_gcp(cat_new, category_prefix, GCS_BUCKET_NAME, S3_BUCKET_NAME)

    new_processed = pd.concat([mixed_new, cat_new], ignore_index=True)

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