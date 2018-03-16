## Introduction

#### What Will I Learn?
In this tutorial we will download the [Special Database 19](https://www.nist.gov/srd/nist-special-database-19), then sort
the images and rename them. After that, we use OpenCV to read the images as a numpy array and scale them down to 32x32
and convert them to greyscale. We then saved these arrays to disk.

#### Requirements

- Python 3
- OpenCV for Python (I downloaded it from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv))

#### Difficulty
- Intermediate

### References
All the files and scripts presented are available on my GitHub Page [here](https://github.com/frereit/TensorflowHandwritingRecognition).
This tutorial is part of a series. I will explain in detail how I achieved the handwriting recognition engine with 
tensorflow. This is the first part, so look out for more!

##So, let's get started!

### 1. Downloading the data
In this example, I will use the [Special Database 19](https://www.nist.gov/srd/nist-special-database-19) published by 
the [National Institute for Standards And Technology](https://www.nist.gov/). It contains over 800,000 pre-classified 
images of handwritten letters and digits. It differentiates between 47 classes: All uppercase letters, all numbers and a 
few lower case letters. I downloaded the `by_merge.zip` file and saved in in my projects folder. 

### 2. Preparing the data for conversion
The database contains over 800,000 images. That's a bit much for my purpose, because the more images we have, the longer 
the training process will take later. About 100,000 images should be enough. To make working with the files easier, I 
wrote [a python script](https://github.com/frereit/TensorflowHandwritingRecognition/blob/master/rename_images.py) to 
move 1/8th the images into the same folder and rename them `class_[class]_index_[index].png`, for example 
`class_25_Index_3743.png`.
```python
def get_class(str):
    return str.split("\\")[1]
```
Simply get the class from a file path. E.g.:
```python
>>> get_class(r"./by_merge\4e\hsf_3\hsf_3_00002.png")
'4e'
```
The `get_class()` function as to be modified if you change `by_merge_dir`, because the path might look different.\
```python
by_merge_dir = "./by_merge"
output_dir = "./data/"
index = 0
class_index = -1
counter = 0
n_copied = 0
classes = []
``` 
`by_merge_dir` and `output dir` are self-explanatory.\
`index` is a variable to keep track of the number of images in a class. This is used to guarantee unique file names.\
`class_index` is the class that is currently being processed. Every time we start traversing a new class folder, this 
variable will be increased.\
`counter` is used to keep track of the number of files traversed.\
`n_copied` is used to keep track of the number of copied files.\
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
if counter % 8 == 0 and file.endswith(".png"):
```
Everything after this if `counter` is divisible by 8, essentially only copying every 8th file and we are dealing with a
png file.

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

```python
counter += 1
```
Lastly, increase the counter.

```python
print("Total images: " + str(n_copied))
```
When we're done with the script, print out how many images we're copied.

And that's it. [Here](https://github.com/frereit/TensorflowHandwritingRecognition/blob/master/rename_images.py)'s the 
full script. This script could take some time to complete, so be patient. 

### Converting the data to numpy arrays

We will now convert the images to numpy arrays. First we will read the single image as an array with the shape \[32,32,1]
We will then put this array into an array with the shape \[101784, 32, 32, 1]. For this I used 
[data_handler.py](https://github.com/frereit/TensorflowHandwritingRecognition/blob/master/data_handler.py).

```python
def get_2d_array(im_path, shape):
    im_color = cv2.imread(im_path)
```
First I use cv2 to get the image as a numpy array. The images have 64x64 pixels, so `cv2.imread` will return a numpy
array with the shape \[64,64,3]: \[x_pixels, y_pixels, color_channels]. 
```python
im_color = cv2.resize(im_color, (32,32))
```
I then resize the image to 32x32, because for handwriting 32x32 is enough. 
```python
im = np.zeros(shape=(32,32,1))
```
Since cv2.imread returns a color image, but we want greyscale, so we'll have to take the average of all color channels
and turn it into an array with the shape \[32,32,1]. 

```python
for i, x in enumerate(im_color):  # Fill the array
    for n, y in enumerate(x):  # Note: We cannot use cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY), because
        im[i][n][0] = (y[0] + y[1] + y[2]) / 3
```
Fill the greyscale array with the averages of the 3 color channels.\
`for i, x in enumerate(im_color):` `i` is an index of the loop, `x` is the object at the position of the index in the 
parameter array.\
`im[i][n][0] = (y[0] + y[1] + y[2]) / 3`. Put the average öf tbe 3 color channels into the final array.

```python
return im 
```
That's all for the `get_2d_array` function. 

```python
def get_label(name):
    return int(name.split("_")[1])
```
Just a helper function to get the class from a filename. E.g.: 
```python
>>> get_label("class_10_index_3454.jpeg")
10
```

```python
n_labels = 47
n_images = 101784
path = "./data"
```
This should be self-explanatory.

```python
images = np.zeros(shape=(n_images, 32, 32, 1))
labels = np.zeros(shape=(n_images, n_labels))
```
These are the array that will contain our database. `images` will contain the image data and `labels` the classes. 
`labels` will be a one-hot-encoded array, thus having as many rows as we have labels. The label for class 3 will look 
like this: \[0, 0, 1, 0, .., 0].  


```python
for i, file in enumerate(os.listdir(path)):
    label = get_label(file)
```
This should be self-explanatory. `enumerate` was explained further up.

```python
image = get_2d_array(os.path.join(path, file))
```
Get the single image. Shape: \[32,32,1]
```python
images[i] = image
labels[i, label] = 1
```
Add the single image to the array of all images. Also one-hot-encode the label into the labels array by setting the
corresponding index to 1. 

```python
print(str(i / n_images * 100) + "% done")
```
Log how far we are. 

```python
np.save("nist_labels_32x32", labels)
np.save("nist_images_32x32", images)
```
When we are done, save the arrays.

And that's it. [Here](https://github.com/frereit/TensorflowHandwritingRecognition/blob/master/data_handler.py)'s the 
full script. And, again, this could take a lot of time to complete, so be patient.

### Recap
In this tutorial we downloaded the [Special Database 19](https://www.nist.gov/srd/nist-special-database-19), then sorted
the images and renamed them. After that, we used OpenCV to read the images as a numpy array and scaled them down to 32x32
and converted them to greyscale. We then saved these arrays to disk.


## Curriculum
Thank you for reading my tutorial! I hope you liked it. If you have any recommendations for future tutorials please
leave a comment. I'll upvote any constructive criticism