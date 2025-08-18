import sys

def convert_gr3d_to_gr2d(input_file, output_file):
    """
    Convert ISPD .gr 3D format to 2D format.
    Keeps only x, y dimensions and layer 1 capacities.
    """
    with open(input_file, "r") as fin:
        lines = fin.readlines()

    out_lines = []
    header_done = False
    num_x, num_y, num_layers = 0, 0, 0

    for line in lines:
        if line.startswith("#"):  # comments
            continue

        parts = line.strip().split()

        # Header line: grid sizes
        if not header_done and len(parts) == 3:
            num_x, num_y, num_layers = map(int, parts)
            out_lines.append(f"{num_x} {num_y} 1\n")  # force 1 layer in 2D
            header_done = True
            continue

        # Capacity line (x, y, layer, capH, capV)
        if len(parts) == 5:
            x, y, layer, capH, capV = parts
            if layer == "1":  # only keep layer 1
                out_lines.append(f"{x} {y} {capH} {capV}\n")
            continue

        # Net lines (netID #pins)
        if len(parts) >= 2 and parts[0].isdigit():
            out_lines.append(line)
            continue

        # Pin lines (x y layer)
        if len(parts) == 3:
            x, y, layer = parts
            out_lines.append(f"{x} {y}\n")  # drop layer info
            continue

        # Route guides etc: drop or keep as-is
        else:
            out_lines.append(line)

    with open(output_file, "w") as fout:
        fout.writelines(out_lines)

    print(f"Converted {input_file} → {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gr3d_to_gr2d.py <input.3d.gr> <output.2d.gr>")
        sys.exit(1)

    convert_gr3d_to_gr2d(sys.argv[1], sys.argv[2])