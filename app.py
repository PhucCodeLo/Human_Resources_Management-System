from flask import Flask, render_template, Response, redirect, url_for
import cv2
from pyngrok import ngrok
import time

app = Flask(__name__)

port_no = 5000
ngrok.set_auth_token('2W0r7lrJB3cS7xPO367SVLJBGzr_347FkBW4xGvtPiBmRkJoq')
public_url = ngrok.connect(port_no).public_url

print(f"App URL: {public_url}")

camera_opened = False

camera = None  # Không khởi tạo camera khi chạy ứng dụng

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

@app.route('/')
def index():
    return render_template('cam.html')

def generate_frames():
    while True:
        if camera_opened:
            success, frame = get_camera().read()
            if not success:
                break
            else:
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera')
def toggle_camera():
    global camera_opened
    camera_opened = not camera_opened
    return redirect(url_for('index'))

@app.route('/page1')
def page1():
    return render_template('index.html')

@app.route('/page2')
def page2():
    return render_template('index2.html')

if __name__ == '__main__':
    app.run(port=port_no, threaded=True)
