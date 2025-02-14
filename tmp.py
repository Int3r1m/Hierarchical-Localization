from hloc import pairs_from_exhaustive
from pathlib import Path

images = Path("/home/dawa/桌面/1F/overlap_0")
output = Path("/home/dawa/桌面/1F/pair_overlap_0.txt")
image_list = [p.relative_to(images).as_posix() for p in (images).iterdir()]

pairs_from_exhaustive.main(output, image_list)