# predict.py

from fastai.vision.all import *
import argparse
import csv
from datetime import datetime
from tqdm import tqdm

# --- Helper functions ---

def denorm_preds(pred, img_size):
    W, H = img_size
    pred_px = torch.zeros_like(pred)
    pred_px[:, 0] = pred[:, 0] * W/2 + W/2
    pred_px[:, 1] = pred[:, 1] * H/2 + H/2
    return pred_px

def keypoint_distance(inp, targ):
    pred = inp.view(-1, 2, 2)
    return ((pred - targ)**2).sum(dim=-1).sqrt().mean()

def keypoint_func(image):
    img = Path(image).stem
    for entry in keypoints_data:
        if (entry['image'] == f"{img}.png"):  # Adjust if needed
            return [[float(x), float(y)] for x, y in entry['keypoints']]
    return [[0.0, 0.0], [0.0, 0.0]]
    
def get_keypoints(img_path):
    "Lookup keypoints by image stem."
    stem = Path(img_path).stem
    for e in keypoints_data:
        if Path(e['image']).stem == stem:
            return [[float(x), float(y)] for x,y in e['keypoints']]
    return [[0.0,0.0],[0.0,0.0]]

def keypoint_distance(preds, targs):
    "Mean Euclidean distance over our two keypoints."
    p = preds.view(-1,2,2)
    return ((p - targs)**2).sum(dim=-1).sqrt().mean()

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input_folder', type=str, required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    backbone = model_path.name.split('-')[1]
    date_str = datetime.now().strftime("%Y%m%d")

    input_folder = Path(args.input_folder)
    output_folder = Path(f'data/predict/{backbone}-{date_str}')
    output_folder.mkdir(parents=True, exist_ok=True)

    logs_folder = Path('data/logs')
    logs_folder.mkdir(parents=True, exist_ok=True)
    log_path = logs_folder / f"predict-log-{backbone}-{date_str}.txt"

    print(f"📦 Loading model {model_path}...")
    learn = load_learner(model_path)

    image_files = get_image_files(input_folder)
    if not image_files:
        print(f"❌ No images found in {input_folder}")
        return

    test_dl = learn.dls.test_dl(image_files)
    preds, _ = learn.get_preds(dl=test_dl)

    csv_path = output_folder / f"kururu-predict-{backbone}-{date_str}.txt"

    with open(csv_path, 'w', newline='') as csvfile, open(log_path, 'w') as logfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['filename', 'x1', 'y1', 'x2', 'y2'])

        for img_path, pred in tqdm(zip(image_files, preds), total=len(image_files), desc="🔮 Predicting"):
            img_name = img_path.name
            pred = pred.view(-1, 2)
            img_obj = PILImage.create(img_path)
            pred_px = denorm_preds(pred, img_obj.size)

            # Save figure with keypoints
            fig, ax = plt.subplots()
            ax.imshow(img_obj)
            ax.scatter(pred_px[:, 0], pred_px[:, 1], color='red', s=20)
            ax.set_title(f"Prediction: {img_name}")
            plt.axis('off')

            save_path = output_folder / f"{img_name}-{backbone}-{date_str}.png"
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

            # Write CSV
            row = [img_name] + pred_px.flatten().tolist()
            writer.writerow(row)

            # Log details
            logfile.write(f"{img_name}\t{pred_px.flatten().tolist()}\n")

    print(f"\n✅ Predictions saved to: {output_folder}")
    print(f"✅ CSV saved to: {csv_path}")
    print(f"✅ Log saved to: {log_path}")

if __name__ == "__main__":
    main()

