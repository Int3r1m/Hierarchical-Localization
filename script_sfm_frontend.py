from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction,
)
from pathlib import Path
from hloc import pairs_from_sequential, pairs_from_exhaustive, match_dense

images = Path("/media/dawa/HIKSEMIHIKSEMIMDC1/WuhanPeoplesParadise/images/4F_Merging")
outputs = Path("/media/dawa/HIKSEMIHIKSEMIMDC1/WuhanPeoplesParadise/SfM/4F_Merging")
# global_features = outputs / "global_features.h5"
local_features = outputs / "local_features.h5"
matches = outputs / "matches.h5"
pairs_sfm = outputs / "pairs.txt"

# retrieval_conf = extract_features.confs["eigenplaces"]
feature_conf = extract_features.confs["aliked-n16"]
matcher_conf = match_features.confs["aliked+lightglue"]
# dense_matcher_conf = match_dense.confs["loftr_superpoint"]

image_list = [p.relative_to(images).as_posix() for p in (images).iterdir()]
# retrieval_path = extract_features.main(
#     retrieval_conf, images, image_list=image_list, feature_path=global_features
# )                          
feature_path = extract_features.main(
    feature_conf, images, image_list=image_list, feature_path=local_features
)

# pairs_from_retrieval.main(retrieval_path, pairs_sfm, num_matched=17)
# pairs_from_sequential.main(images, pairs_sfm, overlap=3)
# pairs_from_exhaustive.main(pairs_sfm, image_list)
match_path = match_features.main(
    matcher_conf, pairs_sfm, "local_features", outputs
)
# feature_path, match_path = match_dense.main(
#     dense_matcher_conf, pairs_sfm, images, outputs, max_kps=4096
# )
model = reconstruction.main(outputs, images, pairs_sfm, feature_path, match_path, skip_geometric_verification=False)
