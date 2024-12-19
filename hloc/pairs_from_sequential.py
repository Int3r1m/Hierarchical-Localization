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
    file_name_list = [f.name for f in path.iterdir() if f.is_file()]
    pairs = []
    for i in range(len(file_name_list)):
        img_name1 = file_name_list[i]
        for j in range(overlap):
            tmp = fibonacci_sequence(j + 1)
            img_name2_fibonacci = file_name_list[(i + tmp) % len(file_name_list)]
            pair_fibonacci = (img_name1, img_name2_fibonacci)
            pairs.append(pair_fibonacci)
    with open(output, "w") as f:
        for pair in pairs:
            f.write(" ".join([pair[0], pair[1]]) + "\n")
    return pairs