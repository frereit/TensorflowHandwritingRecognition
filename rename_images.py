import os
from shutil import copyfile

by_merge_dir = "./by_merge"
output_dir = "./data/"


def get_class(str):
    return str.split("\\")[1]


def main():
    index = 0
    class_index = -1
    classes = []
    for subdir, dirs, files in os.walk(by_merge_dir):
        for file in files:
            if get_class(subdir) not in classes:
                classes.append(get_class(subdir))
                class_index += 1
                index = 0
            copyfile(os.path.join(subdir, file),
                     os.path.join(output_dir, "class_" + str(class_index) + "_index_" + str(index) + ".png"))

            print("Copied " + os.path.join(subdir, file) + " to "
                  + os.path.join(output_dir, "class_" + str(class_index) + "_index_" + str(index) + ".png"))
            index += 1


if __name__ == "__main__":
    main()
