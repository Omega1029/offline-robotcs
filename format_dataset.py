import json, glob
from datasets import Dataset

examples = []
for meta_file in glob.glob("captured_frames/*.json"):
    m = json.load(open(meta_file))
    examples.append({
        "image": m["filename"],
        "caption": m["robot_action"]
    })

ds = Dataset.from_list(examples)
#ds.save_to_disk("robot_action_caption_dataset")