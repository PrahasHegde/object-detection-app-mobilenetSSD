#mobilenetSSD model using flask server

from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Paths to the model files
prototxt = "/home/ubuntu/ros2_ws/Detection/ObjectDet-MobnetSSD/MobileNetSSD_deploy.prototxt"
caffe_model = "/home/ubuntu/ros2_ws/Detection/ObjectDet-MobnetSSD/MobileNetSSD_deploy.caffemodel"

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

# Dictionary with the object class id and names on which the model is trained
classNames = {0: 'background',
              2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
              10: 'cow', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant',
              18: 'sofa'}

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Size of image
        width = frame.shape[1]
        height = frame.shape[0]
        # Construct a blob from the image
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)
        # Blob object is passed as input to the object
        net.setInput(blob)
        # Network prediction
        detections = net.forward()

        # Detections array is in the format 1,1,N,7, where N is the #detected bounding boxes
        # For each detection, the description (7) contains: [image_id, label, conf, x_min, y_min, x_max, y_max]
        for i in range(detections.shape[2]):
            # Confidence of prediction
            confidence = detections[0, 0, i, 2]
            # Set confidence level threshold to filter weak predictions
            if confidence > 0.5:
                # Get class id
                class_id = int(detections[0, 0, i, 1])
                # Scale to the frame
                x_top_left = int(detections[0, 0, i, 3] * width)
                y_top_left = int(detections[0, 0, i, 4] * height)
                x_bottom_right = int(detections[0, 0, i, 5] * width)
                y_bottom_right = int(detections[0, 0, i, 6] * height)

                # Draw bounding box around the detected object
                cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0))

                if class_id in classNames:
                    # Get class label
                    label = classNames[class_id] + ": " + str(confidence)
                    # Get width and text of the label string
                    (w, h), t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    y_top_left = max(y_top_left, h)
                    # Draw bounding box around the text
                    cv2.rectangle(frame, (x_top_left, y_top_left - h), (x_top_left + w, y_top_left + t), (0, 0, 0), cv2.FILLED)
                    cv2.putText(frame, label, (x_top_left, y_top_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)