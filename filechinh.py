from flask import Flask, render_template, Response, request, redirect
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

def process_images_from_firebase():
    # Lấy danh sách các tệp tin trong thư mục trên Firebase Storage
    firebase_image_folder = "images"
    firebase_images = storage.bucket().list_blobs(prefix=firebase_image_folder)

    # Mảng để lưu trữ vector mã hóa và tên của mỗi ảnh
    firebase_face_encodings_list = []
    firebase_image_names = []

    # Mã hóa tất cả các ảnh từ Firebase Storage
    for firebase_image_blob in firebase_images:
        if firebase_image_blob.name.endswith(('.jpg', '.png')):
            try:
                image_data = firebase_image_blob.download_as_bytes()
                nparr = np.frombuffer(image_data, np.uint8)
                firebase_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except cv2.error as decode_error:
                print(f"Error decoding image {firebase_image_blob.name}: {decode_error}")
                continue

            firebase_face_locations = face_recognition.face_locations(firebase_image)
            firebase_face_encodings = face_recognition.face_encodings(firebase_image, firebase_face_locations)

            if firebase_face_encodings:
                firebase_face_encodings_list.extend(firebase_face_encodings)
                firebase_image_names.append(firebase_image_blob.name)
        else:
            print(f"Skipping non-image file: {firebase_image_blob.name}")

    return firebase_face_encodings_list, firebase_image_names

def generate_frames():
    firebase_face_encodings_list, firebase_image_names = process_images_from_firebase()

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Mã hóa khuôn mặt từ webcam
        webcam_face_locations = face_recognition.face_locations(frame)
        webcam_face_encodings = face_recognition.face_encodings(frame, webcam_face_locations)

        # Biến boolean để kiểm tra xem có khuôn mặt khớp hay không
        face_matched = False


        # Khởi tạo biến đếm khuôn mặt
        face_count = 0
        # So sánh vector mã hóa từ webcam và từ Firebase
        for i, webcam_encoding in enumerate(webcam_face_encodings):
            for j, firebase_encoding in enumerate(firebase_face_encodings_list):
                result = face_recognition.compare_faces([firebase_encoding], webcam_encoding)
                if result[0]:
                    # Hiển thị hộp xung quanh khuôn mặt
                    top, right, bottom, left = webcam_face_locations[i]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Hiển thị tên dưới hộp
                    font = cv2.FONT_HERSHEY_DUPLEX
                    name = str(firebase_image_names[j]).replace(".jpg", "").replace(".png", "")

                    cv2.putText(frame, name, (left + 6, bottom + 20), font, 0.5, (255, 255, 255), 1)

                    # Tăng biến đếm khuôn mặt lên 1
                    face_count += 1

                    # Đặt biến boolean thành True khi có khuôn mặt khớp
                    face_matched = True

        # Hiển thị số lượng khuôn mặt đã được nhận diện
        cv2.putText(frame, f"Number of Faces: {face_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Nếu không có khuôn mặt khớp, hiển thị "Unknown"
        if not face_matched:
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, "No face or not registered", (10, 30), font, 1, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')   

@app.route('/')
def index():
    return render_template('cam.html')

def crop_face(image):
    # Chuyển đổi ảnh từ định dạng bytes sang mảng numpy
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Xác định vị trí khuôn mặt trong ảnh
    face_locations = face_recognition.face_locations(img)

    if not face_locations:
        return None

    # Lấy vị trí của khuôn mặt đầu tiên
    top, right, bottom, left = face_locations[0]

    # Cắt ảnh chỉ chứa khuôn mặt
    face_image = img[top:bottom, left:right]

    # Chuyển đổi ảnh cắt thành định dạng bytes
    _, buffer = cv2.imencode('.jpg', face_image)
    cropped_image = buffer.tobytes()

    return cropped_image

@app.route('/upload', methods=['POST'])
def upload():
    # Lấy file từ request
    image = request.files['imageUpload'].read()

    # Crop khuôn mặt
    cropped_image = crop_face(image)

    if cropped_image is not None:
        # Đặt tên file trên Firebase Storage
        filename = f'images/{request.files["imageUpload"].filename}'

        # Lưu ảnh đã cắt vào Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob(filename)
        blob.upload_from_string(cropped_image, content_type='image/jpeg')

        return 'Tải ảnh lên Firebase thành công!'

    return 'Không tìm thấy khuôn mặt trong ảnh.'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/page1')
def page1():
    return render_template('index.html')


# Tài khoản admin đơn giản (trong thực tế, sử dụng cơ sở dữ liệu)
admin_account = {'username': 'admin', 'password': 'admin', 'role': 'admin'}

@app.route('/page2', methods=['GET', 'POST'])
def page2():
    error = None

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == admin_account['username'] and password == admin_account['password']:
            # Đăng nhập thành công, thực hiện các hành động sau đăng nhập
            return redirect("https://console.firebase.google.com/project/project-cpv301/storage/project-cpv301.appspot.com/files/~2Fimages")
        
        else:
            error = 'Bạn không có quyền sử dụng chức năng này nếu bạn không phải nhà phát triển'

    return render_template('index2.html', error=error)

if __name__ == '__main__':
    app.run(debug=True)
