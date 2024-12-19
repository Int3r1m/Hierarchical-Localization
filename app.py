from flask import Flask, request, jsonify
app = Flask(__name__)
import torch
import numpy as np
import base64, uuid
def pair_from_global_descs(
    query_global_descs: torch.Tensor,
    refer_global_descs: torch.Tensor,
    query_img_names: list[str],
    refer_img_names: list[str],
    k: int,
) -> list[tuple[str]]:
    scores = torch.einsum("id,jd->ij", query_global_descs.to('cpu'), refer_global_descs.to('cpu'))
    invalid = torch.from_numpy(np.array(query_img_names)[:, None] == np.array(refer_img_names)[None])
    assert scores.shape == invalid.shape

    scores.masked_fill_(invalid.to('cpu'), float("-inf"))
    topk = torch.topk(scores, k, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()

    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    return [(query_img_names[i], refer_img_names[j]) for i, j in pairs]


@app.route('/', methods=['POST'])
def visual_localization():
    time_based_uuid = uuid.uuid1()
    query_img_name = f'{time_based_uuid}.jpg'
    query_img_path = query_imgs_path / query_img_name

    if 'image' not in request.json:
        return jsonify({'Error': 'There is no key called image!'}), 400
    data = request.json['image']
    if data['image'].startswith('data:image'):
        img_data = data['image'].split(',')[1]
    img_binary = base64.b64decode(img_data)
    with open(query_img_path, 'wb') as f:
        f.write(img_binary)

    query_global_feats = extract_features.main_nonpersistence(
        global_feats_config, model_g, query_imgs_path, [query_img_name])
    
    query_local_feats = extract_features.main_nonpersistence(
        local_feats_config, model_l, query_imgs_path, [query_img_name])
    
    pairs = pair_from_global_descs(
        query_global_feats[query_img_name]["global_descriptor"], refer_global_descs, [query_img_name], refer_img_names, k=7)
    
    matches = match_features.match_from_paths_nonpersistence(
        match_config, pairs, query_local_feats, args.refer_local_feats_path)
    
    options = pycolmap.ImageReaderOptions()
    options.camera_model = "PINHOLE"
    camera = pycolmap.infer_camera_from_image(query_img_path, options)

    config = {
        "estimation": {"ransac": {"max_error": 9}},
        "refinement": {"refine_focal_length": True, "refine_extra_params": True},
    }
    localizer = QueryLocalizer(refer_reconstruction, config)

    refers = []
    for pair in pairs:
        refers.append(pair[1])
    refer_idx = [refer_reconstruction.find_image_with_name(refer).image_id for refer in refers]
    ####################################################################################################################
    clusters = do_covisibility_clustering(refer_idx, refer_reconstruction)
    best_inliers = 0
    best_cluster = None
    logs_clusters = []
    for i, cluster_ids in enumerate(clusters):
        ret, log = pose_from_cluster_nonpersistence(
            localizer, query_img_name, camera, cluster_ids, query_local_feats, matches
        )
        if ret is not None and ret["num_inliers"] > best_inliers:
            best_cluster = i
            best_inliers = ret["num_inliers"]
        logs_clusters.append(log)
    if best_cluster is not None:
        ret = logs_clusters[best_cluster]["PnP_ret"]
        query_img_pose = pycolmap.Image(cam_from_world=ret["cam_from_world"])
    ####################################################################################################################

    # ret, _ = pose_from_cluster_nonpersistence(localizer, query_img_name, camera, refer_idx, query_local_feats, matches)
    # query_img_pose = pycolmap.Image(cam_from_world=ret["cam_from_world"])
    Path.unlink(query_img_path)
    return {'world_from_cam': query_img_pose.cam_from_world.inverse().matrix().tolist(),
            'cam_from_world': query_img_pose.cam_from_world.matrix().tolist(),
            'num_inliers': ret['num_inliers'],
            'inliers_ratio': ret['num_inliers'] / len(ret['inlier_mask'])}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    '''-----Service Parameters-----'''
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='IP address of the service running.')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port of the service running.')
    parser.add_argument('--threads', type=int, default=1,
                        help='Threads of the service running.')
    '''-----References Parameters-----'''
    parser.add_argument('--refer_imgs_path', type=str, default='./areas/10F/images',
                        help='Reference images path.')
    parser.add_argument('--refer_global_feats_path', type=str, default='./areas/10F/features/global_feats_retriangulation.h5',
                        help='Reference global features path.')
    parser.add_argument('--refer_local_feats_path', type=str, default='./areas/10F/features/local_feats_retriangulation.h5',
                        help='Reference local features path.')
    parser.add_argument('--refer_reconstruction_path', type=str, default='./areas/10F/sparse_model',
                        help='Reference reconstruction path.')
    '''------------------------------'''
    args = parser.parse_args()

    import h5py
    import pycolmap
    from pathlib import Path
    from hloc import extractors, extract_features, match_features
    from hloc.utils.base_model import dynamic_load
    from hloc.localize_sfm import QueryLocalizer, pose_from_cluster_nonpersistence, do_covisibility_clustering
    global_feats_config = extract_features.confs["eigenplaces"]
    local_feats_config = extract_features.confs["aliked-n16"]
    match_config = match_features.confs["aliked+lightglue"]

    # refererence reconstruction preload.
    refer_reconstruction = pycolmap.Reconstruction(args.refer_reconstruction_path)

    # global features extractor preload.
    Model_g = dynamic_load(extractors, global_feats_config["model"]["name"])
    model_g = Model_g(global_feats_config["model"]).eval()

    # local features extractor preload.
    Model_l = dynamic_load(extractors, local_feats_config["model"]["name"])
    model_l = Model_l(local_feats_config["model"]).eval()

    # reference global descriptors preload.
    refer_img_names = [refer_img_path.relative_to(Path(args.refer_imgs_path)).as_posix()
                    for refer_img_path in Path(args.refer_imgs_path).iterdir()]
    with h5py.File(str(args.refer_global_feats_path), "r", libver="latest") as hf:
        refer_global_descs = torch.from_numpy(np.stack([hf[refer_img_name]['global_descriptor'].__array__()
                                            for refer_img_name in refer_img_names], 0)).to(torch.float32)

    query_imgs_path = Path.cwd() / "queries"
    query_imgs_path.mkdir(parents=True, exist_ok=True)

    from waitress import serve
    serve(app, host=args.host, port=args.port, threads=args.threads)