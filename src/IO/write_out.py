from pathlib import Path

def write_output(result, relative_path):
    dir = Path(__file__).parent
    absolute_path = dir.joinpath(relative_path)
    with open(absolute_path, "w") as f:
        f.write("id,label\n")
        for i in range(result.shape[0]):
            f.write("{id},{label}\n".format(id=i, label=int(result[i,0])))