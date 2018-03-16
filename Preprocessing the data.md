### 1. Downloading the data
In this example, I will use the [Speical Database 19](https://www.nist.gov/srd/nist-special-database-19) published by 
the [National Institute for Standards And Technology](https://www.nist.gov/). It contains over 800,000 pre-classified 
images of handwritten letters and digits. It differentiates between 47 classes: All uppercase letters, all numbers and a 
few lower case letters. I downloaded the `by_merge.zip` file and saved in in my projects folder. 

### 2. Preparing the data for conversion
To make working with 
the files easier, I wrote [a python script](https://github.com/frereit/TensorflowHandwritingRecognition/blob/master/rename_images.py) 
to move all the images into the same folder and rename them `class_[class]_index_[index].png`, for example 
`class_25_Index_3743.png`.
```python
def get_class(str):
    return str.split("\\")[1]
```
Simply get the class from a file path. E.g.:
```python
>>> get_class(r"./by_merge\4e\hsf_3\hsf_3_00002.png")
4e
```
The `get_class()` function as to be modified if you change `by_merge_dir`, because the path might look different.\
```python
by_merge_dir = "./by_merge"
output_dir = "./data/"
index = 0
class_index = -1
classes = []
``` 
`by_merge_dir` and `output dir` are self-explanatory.\
`index` is a variable to keep track of the number of images in a class. This is used to guarantee unique file names.\
`class_index` is the class that is currently being processed. Every time we start traversing a new class folder, this 
variable will be increased.\
`classes` is a list of all the folder names. If a new one is encountered `class_index` is increased.

```python
for subdir, dirs, files in os.walk(by_merge_dir):
    for file in files:
```
This loops through all files including files in subdirectories.
```python
if get_class(subdir) not in classes:
    classes.append(get_class(subdir))
    class_index += 1
    index = 0
```
If we have not seen this class yet, add it to the list of classes and increase the class index by 1. If  you want you 
can also reset the index to 0, so that the first image of every class has the index 0.

```python
copyfile(os.path.join(subdir, file),
                     os.path.join(output_dir, "class_{}_index_{}.png".format(class_index, index)))
```
Copyfile syntax: `copyfile(src, dst)`\
The source path is just constructed by joining the subdirectory and the file name. \
The destination path is constructed by joining the output directory and the string "class_\[class_index]\_index_\[index].png"
This may not be the fastest possible way to copy a file, but it works for our needs. 

```python
print("Copied " + os.path.join(subdir, file) + " to "
                  + os.path.join(output_dir, "class_{}_index_{}.png".format(class_index, index))
            index += 1
```
Log that we copied a file and increase the index.

And that's it. [Here](https://github.com/frereit/TensorflowHandwritingRecognition/blob/master/rename_images.py)'s the full script. This script could take some time to complete, so be patient. 