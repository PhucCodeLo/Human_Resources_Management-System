import cv2
import numpy as np
from firebase_admin import credentials, storage, initialize_app
import face_recognition

# Khởi tạo Firebase với tệp tin cấu hình của bạn
cred = credentials.Certificate("project-cpv301-firebase-adminsdk-y6l7j-7ea74fc821.json")
initialize_app(cred, {"storageBucket": "project-cpv301.appspot.com"})

# Tên của tệp tin ảnh trong Firebase Storage
firebase_image_filename = "phucvjppro.png"
firebase_image_path = f"images/{firebase_image_filename}"  # Thay đổi đường dẫn tùy thuộc vào cấu trúc thư mục của bạn

# Tải ảnh đã mã hóa từ Firebase Storage
blob = storage.bucket().blob(firebase_image_path)
image_data = blob.download_as_bytes()
nparr = np.frombuffer(image_data, np.uint8)
firebase_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Mã hóa khuôn mặt từ ảnh trong Firebase
firebase_face_locations = face_recognition.face_locations(firebase_image)
firebase_face_encodings = face_recognition.face_encodings(firebase_image, firebase_face_locations)

# Khởi tạo camera
video_capture = cv2.VideoCapture(0)

while True:
    # Đọc frame từ webcam
    ret, frame = video_capture.read()

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
                # name = str(firebase_image_filename)
                name = str(firebase_image_filename).replace(".jpg", "").replace(".png", "")
                cv2.putText(frame, name, (left + 6, bottom + 20), font, 0.5, (255, 255, 255), 1)

                # Đặt biến boolean thành True khi có khuôn mặt khớp
                face_matched = True

    # Nếu không có khuôn mặt khớp, hiển thị "Unknown"
    if not face_matched:
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "Unknown", (10, 30), font, 1, (0, 0, 255), 2)

    # Hiển thị frame từ webcam
    cv2.imshow('Video', frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
video_capture.release()
cv2.destroyAllWindows()



