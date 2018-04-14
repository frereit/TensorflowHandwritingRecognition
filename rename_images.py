import os
from shutil import copyfile


def get_class(str):
    return str.split("\\")[1]


def main():
    by_merge_dir = "./by_merge"
    output_dir = "./data/"
    index = 0
    class_index = -1
    counter = 0
    n_copied = 0
    classes = []
    for subdir, dirs, files in os.walk(by_merge_dir):
        for file in files:
            if get_class(subdir) not in classes:
                classes.append(get_class(subdir))
                class_index += 1
                index = 0
            if counter % 8 == 0 and file.endswith(".png"):
                copyfile(os.path.join(subdir, file),
                        os.path.join(output_dir, "class_{}_index_{}.png".format(class_index, index)))

                print("Copied " + os.path.join(subdir, file) + " to "
                      + os.path.join(output_dir, "class_{}_index_{}.png".format(class_index, index)))
                index += 1
                n_copied += 1
            counter += 1
    print("Total images: " + str(n_copied))

if __name__ == "__main__":
    main()
