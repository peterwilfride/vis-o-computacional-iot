# Usage:
# python test-lpr-api.py -i images

# import required packages
from imutils import paths, resize
import requests
import argparse
from os import path
import cv2

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', required=True,
                help='path to input directory of images')
args = ap.parse_args()

requests.packages.urllib3.disable_warnings()
# IP interno: 10.7.41.14
url = "https://alpr.imd.ufrn.br/lpr/frame"
headers = {
  'Authorization': 'Api-Key E54bdrOv.6lyQSYuvg1lMuhkLD8QAdZSUDWpaKUai'
}

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
    name = imagePath.split(path.sep)[-1]
    print("[INFO] processing image {}/{} ({}): ".format(i + 1, totalImages, name), end="")

    # read input image
    image = cv2.imread(imagePath)
    file = {'image': open(imagePath, 'rb')}
    res = requests.post(url, files=file, headers=headers, verify=False)
    if res.status_code != 201:
        print("no plate found!")
    else:
        data = res.json()
        if data['results'][0]['plate'] == "":
            print("no plate found!")
        else:
            print("\n\tProcessing time:\t{:.3f}s".format(data['processing_time']))
            for j in range(len(data['results'])):
                print("\tPlate recognized:\t{} ({:.1f}%)".format(data['results'][j]['plate'],
                                                                 data['results'][j]['score']*100))
                print("\tPlate class:\t\t{}".format(data['results'][j]['class']))
                print("\tPlate candidates:", end= '')
                for candidate in data['results'][j]['candidates']:
                    print(" {} ({:.1f}%)".format(candidate['plate'], candidate['score']*100), end='')
                print("")
                (x, y, w, h) = data['results'][j]['car_bb']
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                (x, y, w, h) = data['results'][j]['plate_bb']
                plate = image[y:y+h, x:x+w].copy()
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
                for k in range(len(data['results'][j]['characters'])):
                    (cx, cy, cw, ch) = data['results'][j]['characters'][k]
                    cx = cx - x
                    cy = cy - y
                    cv2.rectangle(plate, (cx, cy), (cx + cw, cy + ch), (255, 0, 255), 1)
                plate = resize(plate, width=300)
                cv2.imshow(data['results'][j]['plate'], plate)
        
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        
