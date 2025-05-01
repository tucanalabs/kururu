import pandas as pd
import shutil
from pathlib import Path
import json
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Prepare Kururu dataset.")
    parser.add_argument('--excel_file', type=str, required=True, help="Path to the Excel file with annotations.")
    parser.add_argument('--input_folders', type=str, nargs='+', required=True, help="List of input folders containing images.")
    parser.add_argument('--output_folder', type=str, default="data", help="Base output folder for dataset.")
    args = parser.parse_args()

    excel_file = Path(args.excel_file)
    input_folders = [Path(folder) for folder in args.input_folders]
    output_folder = Path(args.output_folder)

    output_json_folder = output_folder / 'train/labels'
    output_train_images_folder = output_folder / 'train/images'
    output_test_images_folder = output_folder / 'test/images'

    output_json_folder.mkdir(parents=True, exist_ok=True)
    output_train_images_folder.mkdir(parents=True, exist_ok=True)
    output_test_images_folder.mkdir(parents=True, exist_ok=True)

    print(f"📖 Reading Excel file: {excel_file}")
    df = pd.read_excel(excel_file)

    # Build JSON
    data_entries = []
    filenames_in_excel = set()
    for _, row in df.iterrows():
        filename = str(row['image']).strip()
        filenames_in_excel.add(filename)
        rostro = eval(row['rostro'])
        cloaca = eval(row['cloaca'])
        data_entries.append({
            "image": filename,
            "keypoints": [rostro, cloaca]
        })

    date_str = datetime.now().strftime("%Y%m%d")
    json_output_path = output_json_folder / f"data_{date_str}.json"
    with open(json_output_path, 'w') as f:
        json.dump(data_entries, f, indent=2)
    print(f"✅ {len(data_entries)} entries written to {json_output_path}")

    # Copy images: training vs test
    found_train = 0
    found_test  = 0

    # Pre-mark any already-copied training images so we don't double-count
    already_existing = {p.stem for p in output_train_images_folder.glob("*") if p.is_file()}
    all_images_found  = set(already_existing)

    for src_folder in input_folders:
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            for img_path in src_folder.rglob(ext):
                stem = img_path.stem
                if stem in all_images_found:
                    continue
                all_images_found.add(stem)

                if stem in filenames_in_excel:
                    dst = output_train_images_folder / img_path.name
                    if not dst.exists(): shutil.copy(img_path, dst)
                    found_train += 1
                else:
                    dst = output_test_images_folder / img_path.name
                    if not dst.exists(): shutil.copy(img_path, dst)
                    found_test += 1

    # Calculate which ones never showed up
    missing_stems = filenames_in_excel - all_images_found

    print(f"✅ {found_train} new training images copied to {output_train_images_folder}")
    print(f"✅ {found_test} new unlabeled test images copied to {output_test_images_folder}")

    if missing_stems:
        print(f"⚠️ {len(missing_stems)} images listed in Excel but not found:")
        for stem in sorted(missing_stems):
            print(f"    • {stem}")

if __name__ == "__main__":
    main()

