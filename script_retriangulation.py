from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    triangulation,
)
from pathlib import Path


def main():
    retriangulation_path = Path(args.retriangulation_path)

    retrieval_config = extract_features.confs["eigenplaces"]
    feat_config = extract_features.confs["aliked-n16"]
    match_config = match_features.confs["aliked+lightglue"]

    pairs = retriangulation_path / "pairs_retriangulation.txt"
    global_feats = retriangulation_path / "global_feats_retriangulation.h5"
    local_feats = retriangulation_path / "local_feats_retriangulation.h5"
    imgs_path = retriangulation_path / 'clip_imgs'
    refer_cam_pose = retriangulation_path / "clip_sparse"

    img_list = [img_path.relative_to(imgs_path).as_posix() for img_path in (imgs_path).iterdir()]

    retrieval_path = extract_features.main(
        retrieval_config, imgs_path, image_list=img_list, feature_path=global_feats
    )

    pairs_from_retrieval.main(retrieval_path, pairs, num_matched=23)   

    feats_path = extract_features.main(
        feat_config, imgs_path, image_list=img_list, feature_path=local_feats
    )

    match_retriangulation = match_features.main(
        match_config, pairs, "local_feats_retriangulation", retriangulation_path
    )

    triangulation.main(
        retriangulation_path, refer_cam_pose, imgs_path, pairs, feats_path, match_retriangulation
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriangulation_path', type=str, default='/media/dawa/HIKSEMIHIKSEMIMDC1/test/datasets/10F/retriangulation2',
                        help='Output path.')
    args = parser.parse_args()
    main()