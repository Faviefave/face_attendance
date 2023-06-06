import math
import cv2
import face_recognition as fr
import time
from os.path import exists
import os
from sklearn import neighbors
import pickle
import numpy as np
from PIL import Image



cascPath = 'haarcascade_frontalface_dataset.xml'  # dataset
faceCascade = cv2.CascadeClassifier(cascPath)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

#embedder = FaceNet()
knn_clf = pickle.load(open("knn_model.clf", 'rb'))
distance_threshold=0.5

video_capture = cv2.VideoCapture(0)  # 0 for web camera live stream
#  for cctv camera'rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'
#  example of cctv or rtsp: 'rtsp://mamun:123456@101.134.16.117:554/user=mamun_password=123456_channel=1_stream=0.sdp'

capture_duration = 60

def train_image_classifier(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    
    X = []
    y = []
    total = 0
    if exists (model_save_path):
        return None
    else:

        # Loop through each person in the training set

        for root, dirs, files in os.walk(train_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    img_path = os.path.join(root,file)
                    label = os.path.basename(os.path.dirname(img_path)).replace(" ","-").lower()
            # Loop through each training image for the current person
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    img = cv2.resize(img,(0, 0), fx=0.5, fy=0.5)
                    #image = fr.load_image_file(img_path)
                    face_bounding_boxes = fr.face_locations(img)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    
                    X.append(fr.face_encodings(img, known_face_locations=face_bounding_boxes)[0])
                    y.append(label)
                    print('Image of {} with file name:{} is processed'.format(label,img_path))
                    total +=1

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)
    
        # Create and train the KNN classifier
        print('Training the KNN Classifier with {} images'.format(total))
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)

        return knn_clf

def camera_stream(model_path, video_capture):
     
     
     process_this_frame = 29

     start_time = time.time()
     # Capture frame-by-frame
     while True:
        ret, frame = video_capture.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # faces = faceCascade.detectMultiScale(
        #     gray,
        #     scaleFactor=1.1,
        #     minNeighbors=5,
        #     minSize=(30, 30),
        #     flags=cv2.CASCADE_SCALE_IMAGE
        # )
        
        #frame = cv2.resize(frame, (0,0),fx=0.8, fy=0.8)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #rgb_frame = frame[:,:, ::-1]
        #rgb_frame = frame
        
        #detect = detector(rgb_frame)
        detections = fr.face_locations(rgb_frame, model="hog")
        #print("face detected", len(detections))

        if len(detections) == 0:
            continue
        else:
            #print("face detected", detections)
            #if process_this_frame % 30 == 0:
                
            embeddings = fr.face_encodings(rgb_frame, known_face_locations=detections)
            #print('embeddings: ', len(embeddings))

            #reshaped_embeddings = np.reshape(embeddings, (1, -1))

            closest_distances = knn_clf.kneighbors(embeddings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(detections))]
            prediction = [(pred, loc) if rec else ("Unknown", loc) for pred, loc, rec in zip(knn_clf.predict(embeddings), detections, are_matches)]

            # Draw a rectangle around the faces
            for name, (top, right, bottom, left ) in prediction:
                # top *= 2
                # right *= 2
                # bottom *= 2
                # left *= 2
                
                #x, y, w, h = rect_to_bb(rect)

                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom-35), (right,bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                # cv2.imshow('Detected Face', frame)
                # if ord('q') == cv2.waitKey(10):
                #     video_capture.release()
                #     cv2.destroyAllWindows()
                #     exit(0)

        

        # Display the resulting frame in browser
        return cv2.imencode('.jpg', frame)[1].tobytes()
        # cv2.imshow('Detected Face', frame)
        # if ord('q') == cv2.waitKey(10):
        #     video_capture.release()
        #     cv2.destroyAllWindows()
        #     exit(0)


#classifier = train_image_classifier(image_dir, model_save_path="knn_model.clf", n_neighbors=2)
#camera_stream("knn_model.clf")

def detectFramefromWeb(frame):
    
    detections = fr.face_locations(frame, model="hog")

    if len(detections) == 0:
        return frame, None
    else:
            
        embeddings = fr.face_encodings(frame, known_face_locations=detections)
   

        closest_distances = knn_clf.kneighbors(embeddings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(detections))]
        prediction = [(pred, loc) if rec else ("Unknown", loc) for pred, loc, rec in zip(knn_clf.predict(embeddings), detections, are_matches)]

        # Draw a rectangle around the faces
        for name, (top, right, bottom, left ) in prediction:
  
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom-35), (right,bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        return frame, name
    