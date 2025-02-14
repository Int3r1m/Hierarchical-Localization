from pathlib import Path

def fibonacci_sequence(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 1
    
    sequence = [1, 1]
    for i in range(2, n):
        next_value = sequence[-1] + sequence[-2]
        sequence.append(next_value)
    
    return sequence[i]

def main(path: Path, output, overlap):
    file_name_list = sorted(
        [f.name for f in path.iterdir() if f.is_file()],
        key=lambda name: int(name.split('_')[-1].split('.')[0])
    )
    print(file_name_list)
    pairs = []
    for i in range(len(file_name_list)):
        img_name1 = file_name_list[i]
        for j in range(overlap):
            img_name2_fibonacci = file_name_list[(i + j + 1) % len(file_name_list)]
            pair_fibonacci = (img_name1, img_name2_fibonacci)
            pairs.append(pair_fibonacci)
    with open(output, "w") as f:
            f.write("\n".join(" ".join([i, j]) for i, j in pairs))
    return pairs

if __name__ == "__main__":
    path = Path("/home/dawa/桌面/1F/images/2")
    output = Path("/home/dawa/桌面/1F/pairs_2.txt")
    main(path, output, 8)