from flask import Flask, render_template, request, jsonify, redirect, send_file
import cv2
import numpy as np
import base64
import face_recognition
import firebase_admin
from firebase_admin import credentials, storage
import os
from flask_caching import Cache
from datetime import datetime
import csv

app = Flask(__name__)

# Initialize Flask-Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Initialize Firebase Admin SDK
cred = credentials.Certificate("project-cpv301-firebase-adminsdk-y6l7j-7ea74fc821.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'project-cpv301.appspot.com'})

@cache.memoize(timeout=3600)
def process_all_faces_from_firebase():
    # Function to process all faces from Firebase storage
    # And extract face encodings along with image names

    firebase_images = storage.bucket().list_blobs(prefix="faces/")  # List objects in directory "faces/"
   
    firebase_face_encodings_list = []
    firebase_image_names = []

    # Get a list of subfolders in "faces/"
    subdirectories = set(os.path.dirname(blob.name) for blob in firebase_images)

    for subdirectory in subdirectories: ## Iterate through each subdirectory in "faces/"
        for firebase_image_blob in storage.bucket().list_blobs(prefix=subdirectory): # Iterate through each image blob in the current subdirectory
            if firebase_image_blob.name.endswith(('.jpg', '.png')): # Check if the image has a valid format (jpg or png)
                try:
                    # Download image data as bytes from Firebase storage
                    image_data = firebase_image_blob.download_as_bytes()

                    # Convert image data bytes to NumPy array
                    nparr = np.frombuffer(image_data, np.uint8)

                    # Decode the NumPy array into an OpenCV image
                    firebase_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                     # Face recognition processing
                    firebase_face_locations = face_recognition.face_locations(firebase_image, model='hog')
                    # Extract face encodings from the image
                    firebase_face_encodings = face_recognition.face_encodings(firebase_image, firebase_face_locations)
                    
                    if firebase_face_encodings:
                        # If face encodings are found, extend the list of face encodings
                        firebase_face_encodings_list.extend(firebase_face_encodings)
                        # Get the folder name from the image path
                        folder_name = os.path.dirname(firebase_image_blob.name)
                        # Append the folder name to the list of image names
                        firebase_image_names.append(folder_name)
                except (cv2.error, Exception) as error:
                     # Handle errors that may occur during image processing
                    print(f"Error processing image {firebase_image_blob.name}: {error}")
                    continue

    # Return the list of face encodings and image names               
    return firebase_face_encodings_list, firebase_image_names



# Function to log face recognition to CSV
def log_face_recognition(name):
    # Log face recognition to a CSV file with timestamp

    now = datetime.now()  # Get the current date and time
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S") # # Format the timestamp as "YYYY-MM-DD HH:MM:SS"

    # Open the CSV file in append mode
    with open('face_recognition_log.csv', mode='a', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write a new row with the face name and timestamp
        writer.writerow([name, timestamp])

    # Check the number of lines in the file
    line_count = sum(1 for line in open('face_recognition_log.csv'))
    
    # If the limit of 200 is reached, reset the file
    if line_count >= 200:
        reset_face_recognition_log()

def reset_face_recognition_log():
    # Function to reset the face recognition log file

    # Create a new empty log file
    with open('face_recognition_log.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Timestamp'])

    print('Face recognition log reset.')


# Route to display UI when running the app
@app.route('/')
def index():
    return render_template('UI.html')

# Route to compare webcam image with database and perform face recognition
@app.route('/recognize', methods=['POST'])
def recognize():
    # Receive image data from the webcam, compare with database, and perform face recognition

    # Extract image data from the POST request and decode it
    image_data = request.form['image_data'].split(",")[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)

    # Decode the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Retrieve face encodings and image names from Firebase
    firebase_face_encodings_list, firebase_image_names = process_all_faces_from_firebase()

    # Detect face locations and extract face encodings from the webcam image
    face_locations = face_recognition.face_locations(image, model='hog')  
    face_encodings = face_recognition.face_encodings(image, face_locations)

    not_face = True  # Default is Not Face
    face_accept = False  # Set the default to not accept faces
    name = ""  # The name of the face is accepted
    unknown_face = True  # Set default to unknown face


    # Loop through each face in the webcam image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # Loop through each face in the Firebase database
        for firebase_encoding, firebase_image_name in zip(firebase_face_encodings_list, firebase_image_names):

            # Compare face encodings with a tolerance of 0.4
            result = face_recognition.compare_faces([firebase_encoding], face_encoding, tolerance=0.4)  
            if result[0]:
                 # If a match is found
                not_face = False  # Set to False if a face is detected
                face_accept = True  # Set to True if the face is accepted
                unknown_face = False  # The face is no longer unknown

                # Draw a green bounding box around the recognized face
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

                # Extract the name from the Firebase image path
                name = str(firebase_image_name).split('/')[-1].replace(".jpg", "").replace(".png", "")

                # Display the name on the image
                cv2.putText(image, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                 # Display a welcome message
                cv2.putText(image, f'Wellcome {name}, Your Face Is Accepted', (20, 450), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

                # Log the face recognition
                log_face_recognition(name)

    if not_face:
        # If there is no face, draw "Face Not Detected Or Face Does Not Have Access Permission!"
        cv2.putText(image, "Face Not Detected Or Face Does Not Have Access Permission!", (20, 450), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)

    if unknown_face:
        # If face is unknown, draw bounding box and text "unknown"
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(image, 'Unknown', (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (225, 225, 255), 1)

    # Encode the image to base64 for sending in the response
    _, img_encoded = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Return the result in JSON format
    return jsonify({'result': 'success', 'image_data': img_base64})


@app.route('/download_history', methods=['GET', 'POST'])
def download_history():
    # Route to download face recognition history as CSV

    error = None

    if request.method == 'POST':
        # Check if the request method is POST
        username = request.form['username']
        password = request.form['password']

        # Check if the entered username and password match the admin credentials
        if username == admin_account['username'] and password == admin_account['password']:

            # If credentials match, send the face recognition log file as an attachment
            return send_file('face_recognition_log.csv', as_attachment=True)

        else:
             # If credentials do not match, set an error message
            error = 'Bạn không có quyền sử dụng chức năng này nếu bạn không phải nhà phát triển'

    return render_template('login-history.html', error=error)

    

admin_account = {'username': 'admin', 'password': 'admin', 'role': 'admin'}
@app.route('/face_registration', methods=['GET', 'POST'])
def face_registration():
    # Route for face registration with Firebase storage

    error = None

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == admin_account['username'] and password == admin_account['password']:
            # Successfully logged in, perform the following actions
            return redirect("https://console.firebase.google.com/project/project-cpv301/storage/project-cpv301.appspot.com/files/~2Ffaces")
        
        else:
            error = 'Bạn không có quyền sử dụng chức năng này nếu bạn không phải nhà phát triển'

    return render_template('login.html', error=error)

if __name__ == '__main__':
    app.run(debug=True)