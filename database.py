import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import datetime

# Use a service account.
cred = credentials.Certificate('key/attendance-system-4f9be-firebase-adminsdk-sbdnb-8ebe364030.json')

app = firebase_admin.initialize_app(cred)

db = firestore.client()


class AttendanceRecord:
    def __init__(self, data_dict):
        self.lecturerid = data_dict['lecturerId']
        self.session = data_dict['session']
        self.semester = data_dict['semester']
        self.klass = data_dict['klass']
        self.course = data_dict['course']
        self.students = data_dict['student_list']
        self.date_time = datetime.datetime.now()
    


    def createAttendance(self):
        doc_ref = db.collection('AttendanceRecords').document()
        doc_ref.set({
            "lecturerid": self.lecturerid,
            "session": self.session,
            "semester": self.semester,
            "klass": self.klass,
            "course": self.course,
            "students": self.students,
            "date_time": datetime.datetime.now()

        })

#createAttendance('MAN-P-1491')

    def readAllDatafromDatabase():
        users_ref = db.collection("AttendanceRecords")
        docs = users_ref.stream()
        for doc in docs:
            print(f'{doc.id} =>{doc.to_dict()}')


    def __repr__(self):
        return(
                f"AttendanceRecord (\
                    lecturerid = {self.lecturerid}, \
                    session = {self.session}, \
                    semester = {self.semester}, \
                    klass = {self.klass}, \
                    course = {self.course}, \
                    students = {self.students} \
                    )"
        )

#createAttendance = AttendanceRecord()
#createAttendance.createAttendance("MAN-P-1493", "2021/2022", 2, "computer year5", "computer CPE 512", ["16/EG/CO/891", "16/EG/CO/891", "16/EG/CO/891"])