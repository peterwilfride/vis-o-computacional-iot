# USAGE
# python check_faces.py --dataset dataset [--detection-method {hog|cnn(default)}] [-w 1]
#
# Running on a mac mini core i7 8GB RAM:
# 00h:01m:18s: Percentage of faces found in 218 images using hog: 79.8 %
# 01h:17m:05s: Percentage of faces found in 218 images using cnn: 99.5 %

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-w", "--window", type=int, default=0,
                help="show window with faces found in image")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] checking faces...")
imagePaths = list(paths.list_images(args["images"]))
totalImages = len(imagePaths)
if totalImages == 0:
    print("[ERROR] no images found")
    exit(1)

nofacefound = 0
start = time.time()
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    print("[INFO] processing image {}/{} ({}): ".format(i + 1, totalImages, name),
          end='')

    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb,
                                            model=args["detection_method"])
    if len(boxes) == 0:
        print("no face found!")
        nofacefound += 1
    else:
        print("{} face(s) found".format(len(boxes)))

    # loop over the recognized faces
    for (top, right, bottom, left) in boxes:
        # draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    if args["window"] == 1:
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)

end = time.time()
print("[DONE] {} seconds. Percentage of faces found in {} "
      "images using {}: {:.1f} %".format((end - start), totalImages, args["detection_method"],
                                         (totalImages - nofacefound) / totalImages * 100))
