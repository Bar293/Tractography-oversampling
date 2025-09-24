import os
import glob
import ants
import nibabel as nib
from dipy.io.streamline import load_tractogram, save_tractogram
import numpy as np
import pandas as pd


# -------------------- CONFIGURATION --------------------
input_dir = r"/app/ds003900-download/derivatives/t"   # root folder with subjects
output_dir = r"/app/Tractoinferno/train"              # output folder for tractograms in MNI space
mni_template_path = r"/app/MNI152_T1_1mm.nii.gz"      # MNI template

os.makedirs(output_dir, exist_ok=True)


# -------------------- APPLY TRANSFORMS TO STREAMLINES --------------------
def apply_transform_to_streamlines(streamlines, transform_list):
    """
    Apply a set of transforms (affine, warp, etc.) to streamlines.
    
    Args:
        streamlines: List of streamlines (each streamline is an array of points).
        transform_list: List of ANTs transformations to apply.
    
    Returns:
        new_streamlines: List of transformed streamlines.
    """
    new_streamlines = []
    for sl in streamlines:
        # Convert streamline to DataFrame with columns x, y, z
        df = pd.DataFrame(sl, columns=['x', 'y', 'z'])
        # Apply transformations using ANTs
        pts = ants.apply_transforms_to_points(
            3,  # 3D
            df,
            transform_list
        )
        # Keep only x, y, z as numpy array
        new_streamlines.append(pts[['x', 'y', 'z']].to_numpy())
    return new_streamlines


# -------------------- LOAD MNI TEMPLATE --------------------
mni = ants.image_read(mni_template_path)


# -------------------- PROCESS EACH SUBJECT --------------------
subjects = [d for d in os.listdir(input_dir) if d.startswith("sub-")]

for subject in subjects:
    subj_dir = os.path.join(input_dir, subject)
    t1_path = os.path.join(subj_dir, "anat", f"{subject}__T1w.nii.gz")
    tract_dir = os.path.join(subj_dir, "tractography")

    if not os.path.exists(t1_path) or not os.path.isdir(tract_dir):
        print(f"[WARN] {subject} is missing T1 or tractography folder. Skipping.")
        continue

    # --- Register T1 → MNI (Affine) ---
    print(f"[INFO] Registering {subject} to MNI space (Affine)...")
    t1 = ants.image_read(t1_path)
    reg = ants.registration(
        fixed=mni,
        moving=t1,
        type_of_transform="Affine"  # faster than SyN
    )
    transforms = reg["fwdtransforms"]

    # --- Create subject output folder ---
    subj_out_dir = os.path.join(output_dir, subject)
    os.makedirs(subj_out_dir, exist_ok=True)

    # --- Process tractograms ---
    trk_files = glob.glob(os.path.join(tract_dir, "*.trk"))
    for trk_path in trk_files:
        fname = os.path.basename(trk_path)
        # Example: sub-1006__AF_L.trk → bundle_name = AF_L
        try:
            _, bundle_name = fname.replace(".trk", "").split("__")
        except ValueError:
            print(f"[WARN] Unexpected filename format in {fname}. Skipping.")
            continue

        # Load tractogram
        sft = load_tractogram(trk_path, t1_path, bbox_valid_check=False)

        # Apply transforms
        mni_streamlines = apply_transform_to_streamlines(
            sft.streamlines,
            transforms,
        )

        # Save in MNI space (2mm) using the template as reference
        from dipy.io.stateful_tractogram import StatefulTractogram, Space
        sft_mni = StatefulTractogram(mni_streamlines, mni_template_path, Space.RASMM)
        out_path = os.path.join(subj_out_dir, f"{bundle_name}.trk")
        save_tractogram(sft_mni, out_path, bbox_valid_check=False)
        print(f"[OK] Saved: {out_path}")


# -------------------- EXECUTION --------------------
print("All tractograms have been processed into MNI space (Affine).")
