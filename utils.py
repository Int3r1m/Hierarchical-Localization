from pathlib import Path


def MergeTXTFiles(file_paths, output_file):
    merged_content = ""
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                merged_content += file.read() + "\n"
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到。")
        except Exception as e:
            print(f"读取文件 {file_path} 时发生错误: {e}")
    try:
        with open(output_file, 'w', encoding='utf-8') as output:
            output.write(merged_content)
        print(f"合并内容已成功写入 {output_file}")
    except Exception as e:
        print(f"写入文件 {output_file} 时发生错误: {e}")


if __name__ == "__main__":
    output_path = Path("/home/dawa/桌面/1F")
    file_paths = [p for p in output_path.iterdir() if p.suffix.lower() == '.txt']
    output_file = str(output_path / "pairs.txt")
    MergeTXTFiles(file_paths, output_file)