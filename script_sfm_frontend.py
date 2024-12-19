from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction,
)
from pathlib import Path
from hloc import pairs_from_sequential

images = Path("/media/dawa/HIKSEMIHIKSEMIMDC1/test/datasets/greenplaza_parkingarage/images")
outputs = Path("/media/dawa/HIKSEMIHIKSEMIMDC1/test/datasets/greenplaza_parkingarage/SfM")
global_features = outputs / "global_features.h5"
local_features = outputs / "local_features.h5"
matches = outputs / "matches.h5"
pairs_sfm = outputs / "pairs.txt"

retrieval_conf = extract_features.confs["eigenplaces"]
feature_conf = extract_features.confs["aliked-n16"]
matcher_conf = match_features.confs["aliked+lightglue"]

image_list = [p.relative_to(images).as_posix() for p in (images).iterdir()]
retrieval_path = extract_features.main(
    retrieval_conf, images, image_list=image_list, feature_path=global_features
)                          
feature_path = extract_features.main(
    feature_conf, images, image_list=image_list, feature_path=local_features
)

# pairs_from_retrieval.main(retrieval_path, pairs_sfm, num_matched=5)
pairs_from_sequential.main(images, pairs_sfm, overlap=7)
match_path = match_features.main(
    matcher_conf, pairs_sfm, "local_features", outputs
)
model = reconstruction.main(outputs, images, pairs_sfm, feature_path, match_path, skip_geometric_verification=True)
