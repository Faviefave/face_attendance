from flask import Flask, render_template, Response, request, send_from_directory, abort
from camera import camera_stream
import base64
import os
import cv2
import numpy as np
from flask_cors import CORS, cross_origin
import time
import io
from flask_socketio import SocketIO, emit
from PIL import Image
from camera import detectFramefromWeb
from database import AttendanceRecord






app = Flask(__name__, template_folder='templates')

app.config["SECRET_KEY"] = "secret"
app.config["DEBUG"] = True

socketio = SocketIO(app, cors_allowed_origins= ['http://127.0.0.1:5000'])
CORS(app)

#video_capture = cv2.VideoCapture(0) 
capture_duration = 30

isStartAttendance = False

attendance_data = {"lecturerId":'', "session":'', "semester":'', "klass":'', "course":'', "student_list":[]}
student_list = []


@app.route("/")
# @cross_origin(origins ='*', methods=['GET','POST'], allow_headers=['Content-Type'] )
def home():
    global isStartAttendance
    if isStartAttendance:
        attendance_data["student_list"] = student_list
        createAttendance = AttendanceRecord(attendance_data)
        createAttendance.createAttendance()
        isStartAttendance = False
        student_list.clear()     
    return render_template("index.html")

@socketio.on("connect")
def handle_connect():
    print("Client Connected")

@socketio.on('image')
def image(data_image):
    # decode and convert into image
    b = io.BytesIO(data_image)
    #print(data_image)
    pil_image = Image.open(b).convert('RGB')
    open_cv_image = np.array(pil_image)

    # Convert RGB to BGR
    frame = open_cv_image[:, :, ::-1].copy()
    #frame = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    #cv2.imshow("frame", frame)
    #print(open_cv_image)
    frame, detectedFace = detectFramefromWeb(frame)
    if not detectedFace is None and detectedFace not in student_list:
        if detectedFace != "Unknown":
            student_list.append(detectedFace)
    
    # Process the image frame
    #frame = cv2.flip(frame, 1)
    buff = cv2.imencode('.jpeg', frame)[1]
    response = io.BytesIO(buff).getvalue()

    #emit response to client
    emit('response_back', response)


@app.route("/display_video", methods=["GET", "POST"])
def display_video():
    global isStartAttendance, attendance_data
    lecturerId = request.form.get("lecturerId")
    session = request.form.get("session")
    semester = request.form.get("semester")
    klass = request.form.get("klass")
    course = request.form.get("course")

    attendance_data['lecturerId'] = lecturerId
    attendance_data['session'] = session
    attendance_data['semester'] = semester
    attendance_data['klass'] = klass
    attendance_data['course'] = course
    
    isStartAttendance = True
    return render_template("video_page.html")

# @app.route("/video_page", methods=["POST"])
# def attendace_page():
#     return render_template("webcam.html")


# def gen_frame():
#     """Video streaming generator function."""
#     start_time = time.time()
#     while (int(time.time()-start_time) < capture_duration):
#         frame = camera_stream("knn_model.clf", video_capture)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#          # concate frame one by one and show result
#         if (int(time.time()-start_time) > capture_duration-2):
#             video_capture.release()
#             cv2.destroyAllWindows()
#             exit()



# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Put this in the src attribute of an img tag."""
#     return Response(gen_frame(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')



def base64_to_image(base64_string):
    # Extract the base64 encoded binary data from the input string
    #print(base64_string)
    #base64_data = base64_string.split(",")[1]
    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_string)
    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


# @socketio.on("connect")
# def test_connect():
#     print("Connected")
#     emit ("my response", {"data":"Connected"})

# @socketio.on("image")
# def receive_image(image):
#     # Decode the base64-encoded image data
#     image = base64_to_image(image)
#     #print(image)
#     # Perform image processing using OpenCV
#     #gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     #frame_resized = cv2.resize(gray, (640, 480))

#     # Encode the processed image as a JPEG-encoded base64 string
#     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
#     result, frame_encoded = cv2.imencode(".jpg", image, encode_param)
#     processed_img_data = base64.b64encode(frame_encoded).decode()

#     # Prepend the base64-encoded string with the data URL prefix
#     b64_src = "data:image/jpg;base64,"
#     processed_img_data = b64_src + processed_img_data

#     # Send the processed image back to the client
#     emit("response_back", processed_img_data)

if __name__ == '__main__':
    socketio.run(app, debug=True)