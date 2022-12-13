# Usage:
# python lpr_image.py -i images -c lpr-yolov3-tiny.cfg -w lpr-yolov3-tiny.weights -cl lpr.name

# import required packages
from imutils import paths, resize
import os
import cv2
import argparse
import numpy as np
import pytesseract

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', required=True,
                help='path to input directory of images')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()

# read class names from text file
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(args.weights, args.config)

# specify the target device as the Myriad processor on the NCS
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)   <<< USE MYRIAD
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)      <<< USE CPU

# function to get the output layer names in the architecture
def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected plate with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 3)

    cv2.putText(img, label, (x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


# grab the paths to the input images in our dataset
print("[INFO] checking images directory... ", end='')
imagePaths = list(paths.list_images(args.images))
totalImages = len(imagePaths)
if totalImages == 0:
    print("no images found! Exiting.")
    exit(1)
else:
    print("{} images found.".format(totalImages))

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the image name from the image path
    name = imagePath.split(os.path.sep)[-1]
    print("[INFO] processing image {}/{} ({}): ".format(i + 1, totalImages, name),
          end='')

    # read input image
    img = cv2.imread(imagePath)

    Width = img.shape[1]
    Height = img.shape[0]
    scale = 0.00392

    # create input blob
    blob = cv2.dnn.blobFromImage(
        img, scale, (416, 416), (0, 0, 0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.3
    nms_threshold = 0.4

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    if len(boxes) > 0:
        print('{} plate(s) detected!'.format(len(boxes)))
        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            draw_bounding_box(img, class_ids[i], confidences[i], round(
                x), round(y), round(x + w), round(y + h))
            
            aux_img = img[round(y):round(y+h), round(x):round(x+w)]
            aux_img = resize(aux_img, width=300)

            # Transforma a imagem para tom de cinza
            gray = cv2.cvtColor(aux_img, cv2.COLOR_BGR2GRAY)

            #BinarizaçãoIMG
            _, binary_img = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            phrase = pytesseract.image_to_string(
                cv2.bitwise_not(binary_img),
                config="--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

            phrase = phrase.upper()

            print(phrase)

            cv2.imshow(phrase, aux_img)
    else:
        print('no plate detected!')
    
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

