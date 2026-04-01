"""
NIR Vein Extraction Pipeline
==============================
Pipeline engineered for domain shift resilience using local contrast enhancement, 
illumination flattening (Black-Hat), and optimized morphological filtering.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# CONFIGURATION 

INPUT_FOLDER           = "forearmNIR"
OUTPUT_FOLDER          = "results"

#I use this below as a quick toggle. 
#This can be set to False to batch process the entire 99-image dataset for the final output. 
# If I need to debug, I flip it to True.
RUN_SINGLE_IMAGE       = False 
SINGLE_IMAGE_PATH      = "forearmNIR/00002 .jpg" # When above is set to True, I can use any image and   pipeline



# PIPELINE FUNCTIONS

def load_image(path):
    img = cv2.imread(path.strip())
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def apply_clahe(gray, clip_limit=2.5, tile_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)# Local contrast enhancement. 
    return clahe.apply(gray)


def morph_cleanup(mask, open_k=3, close_k=15):
    # k_open (3x3): Surgically deletes tiny noise specks before they grow.
    # k_close (15x15): Bridges fragmented gaps in the actual veins without creating bubbles.
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    
    opened  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    closed  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
    return closed


def filter_blobs(mask, min_area=400):
    # This wipes out any surviving noise 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


# VISUALISATION

def show_pipeline(imgs, titles, save_path=None):
    cols = len(imgs)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 5))
    fig.suptitle("NIR Vein Extraction Pipeline (Optimized Classical Morphology)", fontsize=15, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap="gray")
        ax.set_title(titles[i], fontsize=12)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved visual to: {save_path}")
    plt.show()

# MAIN PIPELINE

def run_pipeline(image_path, show=True):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0].strip()

    print(f"\nProcessing: {image_path}")
    original = load_image(image_path)

    # 1. Arm Mask: I mask the arm and heavily erode the mask (45x45) to make sure that the pipeline ignores dark boundary shadows and any physical artifacts (like tape).    
    _, arm_mask_raw = cv2.threshold(original, 15, 255, cv2.THRESH_BINARY)
    kernel_arm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    arm_mask = cv2.erode(arm_mask_raw, kernel_arm, iterations=1)

    # 2. CLAHE: I increased the clip limit to 2.5. This is just enough to bring out faint, deep veins without amplifying the background noise.
    enhanced = apply_clahe(original, clip_limit=2.5)

    # 3. Gaussian Blur: Heavy blurring (15x15) is important here. Without this, the morphing operation that follows would pick up skin pores and hair as tiny vein structures.
    smoothed = cv2.GaussianBlur(enhanced, (15, 15), 0)

    # 4. Black-Hat: This is the main vein extractor. It scoops out dark valley features and smoothes the uneven hardware lighting to allow a global threshold to actually work.
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    bh = cv2.morphologyEx(smoothed, cv2.MORPH_BLACKHAT, kernel_bh)

    # 5. Global Threshold: Now that Black-Hat operation has flattened the background to near-zero, a strict global threshold of 9 is enough to isolate the vein perfectly.
    _, binary = cv2.threshold(bh, 9, 255, cv2.THRESH_BINARY)
    
    # Restrict detections strictly to the interior of the arm
    veins_masked = cv2.bitwise_and(binary, binary, mask=arm_mask)

    # 6. Morphological Cleanup: The tight opening (3x3) eliminates any microscopic static, and a moderate closing (15x15) rejoins any broken segments of the actual veins.
    cleaned = morph_cleanup(veins_masked, open_k=3, close_k=15)
    
    # 7. Blob Filter: Finally, I filter out any blobs that are smaller than 400 pixels. Real vein branches are much bigger, and this eliminates any remaining noise.
    cleaned = filter_blobs(cleaned, min_area=400)

    # Saves the final binary mask
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_mask.png"), cleaned)

    if show:
        imgs   = [original, bh, binary, veins_masked, cleaned]
        titles = ["1. Original", "2. Black-Hat", "3. Threshold (Thresh=9)", "4. Masked", "5. Final Mask"]
        show_pipeline(imgs, titles, save_path=os.path.join(OUTPUT_FOLDER, f"{name}_pipeline.png"))

    return {"cleaned": cleaned}

# BATCH

def run_batch():
    exts  = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(INPUT_FOLDER, "**", ext), recursive=True))

    if not paths:
        print(f"No images found in: {INPUT_FOLDER}")
        return

    paths.sort()
    print(f"Found {len(paths)} images. Processing batch...\n")

    for p in paths:
        run_pipeline(p, show=False)
        
    print(f"\nBatch processing complete. All masks saved to '{OUTPUT_FOLDER}/'")


# ENTRY POINT

if __name__ == "__main__":
    if RUN_SINGLE_IMAGE:
        run_pipeline(SINGLE_IMAGE_PATH)
    else:
        run_batch()
