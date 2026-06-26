import os, glob, json, re

# ================================================================
# CONFIG
# ================================================================
data_dir = os.path.join(os.getcwd(), "captured_frames/captured_frames")

# ================================================================
# NATURAL SORT
# ================================================================
def natural_key(path):
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]

# ================================================================
# LOAD AND SORT FILES
# ================================================================
img_files = sorted(glob.glob(os.path.join(data_dir, "*.jpg")), key=natural_key)
json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")), key=natural_key)

if not img_files or not json_files:
    raise RuntimeError(f"Found {len(img_files)} images and {len(json_files)} JSONs in {data_dir}")

if len(img_files) != len(json_files):
    print(f"⚠️ Warning: {len(img_files)} JPGs and {len(json_files)} JSONs. Will pair by index up to the shortest length.")
min_len = min(len(img_files), len(json_files))
img_files, json_files = img_files[:min_len], json_files[:min_len]

# ================================================================
# DISPLAY MATCHES
# ================================================================
for i, (img_path, json_path) in enumerate(zip(img_files, json_files)):
    try:
        with open(json_path, "r") as f:
            meta = json.load(f)
        robot_action = meta.get("robot_action", "❌ Missing 'robot_action'")
        print(f"\n[{i+1}/{min_len}] {os.path.basename(img_path)} → {os.path.basename(json_path)}")
        print(f"  Action: {robot_action}")
        print("  JSON contents:")
        print(json.dumps(meta, indent=4))
    except Exception as e:
        print(f"⚠️ Error reading {json_path}: {e}")
