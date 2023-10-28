# from flask import Flask, render_template, request, jsonify, redirect
# import cv2
# import numpy as np
# import base64
# import face_recognition
# import firebase_admin
# from firebase_admin import credentials, storage
# import os
# from flask_caching import Cache
# from functools import lru_cache


# app = Flask(__name__)

# # Khởi tạo Flask-Caching
# cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# # Khởi tạo Firebase Admin SDK
# cred = credentials.Certificate("project-cpv301-firebase-adminsdk-y6l7j-7ea74fc821.json")
# firebase_admin.initialize_app(cred, {'storageBucket': 'project-cpv301.appspot.com'})
# # firebase_image_folder = "images"

# @cache.memoize(timeout=3600)
# def process_all_faces_from_firebase():
#     firebase_images = storage.bucket().list_blobs(prefix="faces/")  # Liệt kê các đối tượng trong thư mục "faces/"
   
#     firebase_face_encodings_list = []
#     firebase_image_names = []

#     # Lấy danh sách thư mục con trong "faces/"
#     subdirectories = set(os.path.dirname(blob.name) for blob in firebase_images)

#     for subdirectory in subdirectories:
#         for firebase_image_blob in storage.bucket().list_blobs(prefix=subdirectory):
#             if firebase_image_blob.name.endswith(('.jpg', '.png')):
#                 try:
#                     image_data = firebase_image_blob.download_as_bytes()
#                     nparr = np.frombuffer(image_data, np.uint8)
#                     firebase_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
#                     # Thực hiện các xử lý khác như nhận diện khuôn mặt
#                     firebase_face_locations = face_recognition.face_locations(firebase_image, model='hog') 
#                     firebase_face_encodings = face_recognition.face_encodings(firebase_image, firebase_face_locations)
                    
#                     if firebase_face_encodings:
#                         firebase_face_encodings_list.extend(firebase_face_encodings)
#                         # Lấy tên thư mục từ đường dẫn
#                         folder_name = os.path.dirname(firebase_image_blob.name)
#                         firebase_image_names.append(folder_name)
#                 except (cv2.error, Exception) as error:
#                     print(f"Error processing image {firebase_image_blob.name}: {error}")
#                     continue

#     return firebase_face_encodings_list, firebase_image_names


# #Route hiển thị UI khi chạy App
# @app.route('/')
# def index():
#     return render_template('UI.html')

# #Route so sánh ảnh chụp từ webcam với dữ liệu từ Database
# @app.route('/recognize', methods=['POST'])
# def recognize():
#     image_data = request.form['image_data'].split(",")[1]
#     nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     firebase_face_encodings_list, firebase_image_names = process_all_faces_from_firebase()

#     face_locations = face_recognition.face_locations(image, model='hog')  
#     face_encodings = face_recognition.face_encodings(image, face_locations)

#     not_face = True  # Đặt mặc định là Not Face
#     face_accept = False  # Đặt mặc định là không chấp nhận khuôn mặt
#     name = ""  # Tên của khuôn mặt được chấp nhận

#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         for firebase_encoding, firebase_image_name in zip(firebase_face_encodings_list, firebase_image_names):
#             result = face_recognition.compare_faces([firebase_encoding], face_encoding, tolerance=0.4)  
#             if result[0]:
#                 not_face = False  # Nếu có khuôn mặt, đặt thành False
#                 face_accept = True  # Đặt thành True nếu có khuôn mặt được chấp nhận
#                 cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
#                 name = str(firebase_image_name).split('/')[-1].replace(".jpg", "").replace(".png", "")
#                 cv2.putText(image, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
#                 cv2.putText(image, f'Wellcome {name}, Your Face Is Accepted', (20, 450), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

#     if not_face:
#         # Nếu không có khuôn mặt, vẽ "Face Not Detected Or Face Does Not Have Access Permission!" lên ảnh
#         cv2.putText(image, "Face Not Detected Or Face Does Not Have Access Permission!", (20, 450), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)

#     _, img_encoded = cv2.imencode('.jpg', image)
#     img_base64 = base64.b64encode(img_encoded).decode('utf-8')

#     return jsonify({'result': 'success', 'image_data': img_base64})


# admin_account = {'username': 'admin', 'password': 'admin', 'role': 'admin'}
# @app.route('/page1', methods=['GET', 'POST'])
# def page1():
#     error = None

#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']

#         if username == admin_account['username'] and password == admin_account['password']:
#             # Đăng nhập thành công, thực hiện các hành động sau đăng nhập
#             return redirect("https://console.firebase.google.com/project/project-cpv301/storage/project-cpv301.appspot.com/files/~2Ffaces")
        
#         else:
#             error = 'Bạn không có quyền sử dụng chức năng này nếu bạn không phải nhà phát triển'

#     return render_template('login.html', error=error)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify, redirect
import cv2
import numpy as np
import base64
import face_recognition
import firebase_admin
from firebase_admin import credentials, storage
import os
from flask_caching import Cache
from functools import lru_cache
from pyngrok import ngrok


app = Flask(__name__)

port_no = 5000
ngrok.set_auth_token('2W0r7lrJB3cS7xPO367SVLJBGzr_347FkBW4xGvtPiBmRkJoq')
public_url = ngrok.connect(port_no).public_url

print(f"App URL: {public_url}")

# Khởi tạo Flask-Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Khởi tạo Firebase Admin SDK
cred = credentials.Certificate("project-cpv301-firebase-adminsdk-y6l7j-7ea74fc821.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'project-cpv301.appspot.com'})
# firebase_image_folder = "images"

@cache.memoize(timeout=3600)
def process_all_faces_from_firebase():
    firebase_images = storage.bucket().list_blobs(prefix="faces/")  # Liệt kê các đối tượng trong thư mục "faces/"
   
    firebase_face_encodings_list = []
    firebase_image_names = []

    # Lấy danh sách thư mục con trong "faces/"
    subdirectories = set(os.path.dirname(blob.name) for blob in firebase_images)

    for subdirectory in subdirectories:
        for firebase_image_blob in storage.bucket().list_blobs(prefix=subdirectory):
            if firebase_image_blob.name.endswith(('.jpg', '.png')):
                try:
                    image_data = firebase_image_blob.download_as_bytes()
                    nparr = np.frombuffer(image_data, np.uint8)
                    firebase_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Thực hiện các xử lý khác như nhận diện khuôn mặt
                    firebase_face_locations = face_recognition.face_locations(firebase_image, model='hog') 
                    firebase_face_encodings = face_recognition.face_encodings(firebase_image, firebase_face_locations)
                    
                    if firebase_face_encodings:
                        firebase_face_encodings_list.extend(firebase_face_encodings)
                        # Lấy tên thư mục từ đường dẫn
                        folder_name = os.path.dirname(firebase_image_blob.name)
                        firebase_image_names.append(folder_name)
                except (cv2.error, Exception) as error:
                    print(f"Error processing image {firebase_image_blob.name}: {error}")
                    continue

    return firebase_face_encodings_list, firebase_image_names


#Route hiển thị UI khi chạy App
@app.route('/')
def index():
    return render_template('UI.html')

#Route so sánh ảnh chụp từ webcam với dữ liệu từ Database
@app.route('/recognize', methods=['POST'])
def recognize():
    image_data = request.form['image_data'].split(",")[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    firebase_face_encodings_list, firebase_image_names = process_all_faces_from_firebase()

    face_locations = face_recognition.face_locations(image, model='hog')  
    face_encodings = face_recognition.face_encodings(image, face_locations)

    not_face = True  # Đặt mặc định là Not Face
    face_accept = False  # Đặt mặc định là không chấp nhận khuôn mặt
    name = ""  # Tên của khuôn mặt được chấp nhận

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        for firebase_encoding, firebase_image_name in zip(firebase_face_encodings_list, firebase_image_names):
            result = face_recognition.compare_faces([firebase_encoding], face_encoding, tolerance=0.4)  
            if result[0]:
                not_face = False  # Nếu có khuôn mặt, đặt thành False
                face_accept = True  # Đặt thành True nếu có khuôn mặt được chấp nhận
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                name = str(firebase_image_name).split('/')[-1].replace(".jpg", "").replace(".png", "")
                cv2.putText(image, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(image, f'Wellcome {name}, Your Face Is Accepted', (20, 450), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

    if not_face:
        # Nếu không có khuôn mặt, vẽ "Face Not Detected Or Face Does Not Have Access Permission!" lên ảnh
        cv2.putText(image, "Face Not Detected Or Face Does Not Have Access Permission!", (20, 450), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)

    _, img_encoded = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return jsonify({'result': 'success', 'image_data': img_base64})


admin_account = {'username': 'admin', 'password': 'admin', 'role': 'admin'}
@app.route('/page1', methods=['GET', 'POST'])
def page1():
    error = None

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == admin_account['username'] and password == admin_account['password']:
            # Đăng nhập thành công, thực hiện các hành động sau đăng nhập
            return redirect("https://console.firebase.google.com/project/project-cpv301/storage/project-cpv301.appspot.com/files/~2Ffaces")
        
        else:
            error = 'Bạn không có quyền sử dụng chức năng này nếu bạn không phải nhà phát triển'

    return render_template('login.html', error=error)

if __name__ == '__main__':
    app.run(port=port_no, threaded=True)


