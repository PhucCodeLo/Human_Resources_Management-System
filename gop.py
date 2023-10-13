from flask import Flask, render_template, Response, request, jsonify
import cv2
import dlib
import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import face_recognition

app = Flask(__name__)
camera = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

# Khởi tạo Firebase Admin SDK
cred = credentials.Certificate("project-cpv301-firebase-adminsdk-y6l7j-7ea74fc821.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'project-cpv301.appspot.com'})

# Tên của tệp tin ảnh trong Firebase Storage
firebase_image_filename = "mark_zuckerberg.jpg"
firebase_image_path = f"images/{firebase_image_filename}"  

# Tải ảnh đã mã hóa từ Firebase Storage
blob = storage.bucket().blob(firebase_image_path)
image_data = blob.download_as_bytes()
nparr = np.frombuffer(image_data, np.uint8)
firebase_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Mã hóa khuôn mặt từ ảnh trong Firebase
firebase_face_locations = face_recognition.face_locations(firebase_image)
firebase_face_encodings = face_recognition.face_encodings(firebase_image, firebase_face_locations)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Mã hóa khuôn mặt từ webcam
        webcam_face_locations = face_recognition.face_locations(frame)
        webcam_face_encodings = face_recognition.face_encodings(frame, webcam_face_locations)

        # Biến boolean để kiểm tra xem có khuôn mặt khớp hay không
        face_matched = False

        # So sánh vector mã hóa từ webcam và từ Firebase
        for i, webcam_encoding in enumerate(webcam_face_encodings):
            for j, firebase_encoding in enumerate(firebase_face_encodings):
                result = face_recognition.compare_faces([firebase_encoding], webcam_encoding)
                if result[0]:
                    # Hiển thị hộp xung quanh khuôn mặt
                    top, right, bottom, left = webcam_face_locations[i]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Hiển thị tên dưới hộp
                    font = cv2.FONT_HERSHEY_DUPLEX
                    name = str(firebase_image_filename).replace(".jpg", "").replace(".png", "")
                    cv2.putText(frame, name, (left + 6, bottom + 20), font, 0.5, (255, 255, 255), 1)

                    # Đặt biến boolean thành True khi có khuôn mặt khớp
                    face_matched = True

        # Nếu không có khuôn mặt khớp, hiển thị "Unknown"
        if not face_matched:
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, "Unknown", (10, 30), font, 1, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('cam.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Lấy file từ request
    image = request.files['imageUpload']

    # Đặt tên file trên Firebase Storage
    filename = f'images/{image.filename}'  

    # Lưu ảnh vào Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_string(image.read(), content_type=image.content_type)

    return 'Tải ảnh lên Firebase thành công!'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/page1')
def page1():
    return render_template('index.html')

@app.route('/page2')
def page2():
    return render_template('index2.html')


if __name__ == '__main__':
    app.run(debug=True)




