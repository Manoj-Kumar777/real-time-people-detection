import cv2
import imutils
import numpy as np
import argparse

def detect(box):
    object_box_coordinates, weights =  HOGCV.detectMultiScale(box, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    
    person = 1
    for x,y,w,h in object_box_coordinates:
        cv2.rectangle(box, (x,y), (x+w,y+h), (255,255,255), 2)
        cv2.putText(box, f'PERSON {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        person += 1
    
    cv2.putText(box, 'STATUS : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2)
    cv2.putText(box, f'TOTAL PERSONS= {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2)
    cv2.imshow('output', box)

    return box

def detectByPathVideo(path, writer):

    video = cv2.VideoCapture(path)
    check, box = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path.')
        return

    print('Detecting people...')
    while video.isOpened():
        #check is True if reading was successful 
        check, box =  video.read()

        if check:
            box = imutils.resize(box , width=min(800,box.shape[1]))
            box = detect(box)
            
            if writer is not None:
                writer.write(box)
            
            key = cv2.waitKey(1)
            if key== ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()

def detectByCamera(writer):   
    video = cv2.VideoCapture(0)
    print('Detecting people...')

    while True:
        check, box = video.read()

        box = detect(box)
        if writer is not None:
            writer.write(box)

        key = cv2.waitKey(1)
        if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()

def detectByPathImage(path, output_path):
    image = cv2.imread(path)

    image = imutils.resize(image, width = min(800, image.shape[1])) 

    result_image = detect(image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def humanDetector(args):
    image_path = args["image"]
    video_path = args['video']
    if str(args["camera"]) == 'true' : camera = True 
    else : camera = False

    writer = None
    if args['output'] is not None and image_path is None:
        writer = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))

    if camera:
        print('[INFO] Opening Web Cam.')
        detectByCamera(writer)
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(video_path, writer)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args['output'])

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
    args = vars(arg_parse.parse_args())

    return args

if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    args = argsParser()
    humanDetector(args)
