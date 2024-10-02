from ultralytics import YOLO
import cv2, requests, json, time
from firebase_admin import storage, credentials, initialize_app
from io import BytesIO

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
initialize_app(credentials.Certificate("inspector.json"), { "storageBucket": "inspector-f0183.appspot.com" })

model = YOLO("inspector.pt")
data = {
	"class": "name",
	"confidence": 0
}

while True:
	ret, frame = cap.read()
	frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
	cv2.imshow("inspector", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	results = model.predict(source = frame, conf = 0.5)

	for r in results:
		jsondata = json.loads(r.to_json())
		data["class"] = jsondata[0]["name"]
		data["confidence"] = jsondata[0]["confidence"]
		requests.patch("https://inspector-f0183-default-rtdb.firebaseio.com/.json", json = data)

	is_success, buffer = cv2.imencode(".jpg", frame)
	if is_success:
		io_buf = BytesIO(buffer)
		blob = storage.bucket().blob("inspector.jpg")
		blob.upload_from_file(io_buf, content_type = "image/jpeg")
	time.sleep(1)
