import cv2
import pyttsx3
import torch
import openai
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from ultralytics import YOLO
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import os
import speech_recognition as sr
from scipy.spatial.distance import euclidean

#This api key is personal please, not to be used outside this project
openai.api_key = " "

model = YOLO('yolov10s.pt') #(you va use yolov10m.pt or yolov10l which has more paarameters and slower in processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
depth_model.eval()

transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

engine = pyttsx3.init()

# Function to calibrate depth and convert to feet
def get_calibrated_distance(raw_depth_cm, scaling_factor=0.05):
    calibrated_depth_cm = raw_depth_cm * scaling_factor
    distance_ft = calibrated_depth_cm * 0.0328084
    return distance_ft

# Function to merge duplicate objectson different frames
def merge_objects(detected_objects, distance_threshold=50):
    consolidated_objects = []
    for obj in detected_objects:
        found = False
        for cons_obj in consolidated_objects:
            if (obj["object"] == cons_obj["object"] and
                    euclidean(obj["location"], cons_obj["location"]) < distance_threshold):
                cons_obj["location"] = tuple(
                    np.mean([cons_obj["location"], obj["location"]], axis=0)
                )
                cons_obj["distance_ft"] = np.mean([cons_obj["distance_ft"], obj["distance_ft"]])
                found = True
                break
        if not found:
            consolidated_objects.append(obj)
    return consolidated_objects

# Function to listen for voice commands
def listen_for_command(prompt="Please speak your command.", timeout=12):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print(prompt)
    engine.say(prompt)
    engine.runAndWait()
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=timeout)
        command = recognizer.recognize_google(audio).lower()
        print(f"Command received: {command}")
        return command
    except sr.UnknownValueError:
        print("Could not understand the command.")
        engine.say("I didn't understand that. Please try again.")
        engine.runAndWait()
        return None
    except sr.WaitTimeoutError:
        print("No command detected.")
        engine.say("I didn't hear anything. Please try again.")
        engine.runAndWait()
        return None

# Function to record video
def record_video(output_path="recorded_video.avi", max_duration=12):
    engine.say("Recording started.")
    engine.runAndWait()
    print("Recording started. Say 'stop recording' to end early.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    frame_count = 0
    max_frames = fps * max_duration
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_count % (fps * 2) == 0:  # Check every 2 seconds
            try:
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source, timeout=1)
                command = recognizer.recognize_google(audio).lower()
                if "stop recording" in command:
                    print("Stop command received.")
                    break
            except (sr.UnknownValueError, sr.WaitTimeoutError):
                pass

        frame_count += 1
        if frame_count >= max_frames:
            print("Max duration reached. Stopping recording.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

# Process video frames and extract objects
def process_video(video_path):
    engine.say("Processing started.")
    engine.runAndWait()
    print("Processing started.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    object_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (640, 480))
        results = model(resized_frame, stream=True, conf=0.4)

        input_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(input_image)
        input_tensor = transform(input_image).unsqueeze(0).to(device)

        with torch.no_grad():
            depth_map = depth_model(input_tensor)
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=resized_frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        frame_objects = []
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                label = int(box.cls[0])
                class_name = result.names[label]
                conf = float(box.conf[0])

                object_depth = depth_map[y1:y2, x1:x2]
                raw_depth_cm = np.median(object_depth)
                object_distance_ft = get_calibrated_distance(raw_depth_cm)

                # Determine direction based on x-coordinate
                center_x = (x1 + x2) // 2
                if center_x < frame_width * 0.33:
                    direction = "left"
                elif center_x > frame_width * 0.66:
                    direction = "right"
                else:
                    direction = "center"

                frame_objects.append({
                    "object": class_name,
                    "confidence": conf,
                    "distance_ft": object_distance_ft,
                    "location": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "direction": direction
                })

        object_data.extend(frame_objects)

    cap.release()
    return merge_objects(object_data)

# Generate navigation instructions
def generate_navigation_instructions(data, query):
    prompt = f"The user asked: '{query}'.\nHere is the detected object data:\n"
    for obj in data:
        prompt += f"- {obj['object']} is {obj['distance_ft']:.2f} feet to the {obj['direction']}.\n"
    prompt += "Provide step-by-step navigation instructions using terms like forward, left, and right."

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing navigation instructions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

def interaction_loop(objects):
    detected_objects = {obj['object'] for obj in objects}  # Create a set of detected objects
    while True:
        query = listen_for_command("Where do you want to go?")
        if query in ["stop recording", "stop record"]:
            engine.say("Stopping the interaction. Listening for your next command.")
            engine.runAndWait()
            listen_for_next_command()  # Listening for specific phrases after stopping recording
            break
        elif query:
            # Split query into words and check against detected objects
            query_words = query.split()
            matched_objects = [word for word in query_words if word in detected_objects]

            if not matched_objects:
                engine.say(f"The object you mentioned is not detected in this environment.")
                engine.runAndWait()
                engine.say(f"Detected objects are: {', '.join(detected_objects)}.")
                engine.runAndWait()
                engine.say("How can I assist you further?")
                engine.runAndWait()
            else:
                # Assuming that the first matched object is the primary target
                target_object = matched_objects[0]
                instructions = generate_navigation_instructions(objects, target_object)
                engine.say(instructions)
                engine.runAndWait()
                engine.say("How can I assist you further?")
                engine.runAndWait()
        else:
            engine.say("I couldn't hear your query. Try again.")
            engine.runAndWait()

# Function to listen for specific phrases after stopping recording
def confirm_processing():
    attempts = 0
    while attempts < 3:
        command = listen_for_command("I have recorded a video. Should I proceed to process it? Say yes or no.")
        if command in ["yes", "yes yes", "yes yes yes", "yess"]:
            return True
        elif command in ["no", "no no","noo"]:
            return False
        elif command in ["stop recording", "stop record"]:
            engine.say("Stopping the interaction. Listening for your next command.")
            engine.runAndWait()
            listen_for_next_command()
            return None  # Breaking out of the confirmation loop
        else:
            engine.say("I couldn't hear you. Please respond with yes or no.")
            engine.runAndWait()
            attempts += 1
    engine.say("No response detected. Listening for your next command.")
    engine.runAndWait()
    listen_for_next_command()
    return None  # Break out of confirmation loop

def listen_for_seccommand(prompt="Please speak your command.", timeout=60):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print(prompt)
    engine.say(prompt)
    engine.runAndWait()
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=timeout)
        command = recognizer.recognize_google(audio).lower()
        print(f"Command received: {command}")
        return command
    except sr.UnknownValueError:
        print("Could not understand the command.")
        engine.say("")
        engine.runAndWait()
        return None
    except sr.WaitTimeoutError:
        print("No command detected.")
        engine.say("")
        engine.runAndWait()
        return None

def listen_for_next_command():
    LONG_TIMEOUT = 60  # adjust longer if you like sha, you can also adjust in the seccomand function timeout
    while True:
        try:
            command = listen_for_seccommand(" ", timeout=LONG_TIMEOUT)
        except sr.WaitTimeoutError:
            continue  

        if command == "take a rest":
            engine.say("Taking a rest. Goodbye!")
            engine.runAndWait()
            exit()
        elif command in ["baby i am home", "baby are you there"]:
            engine.say("I am always here to assist you.")
            engine.runAndWait()
            engine.say("Say 'start recording video' to begin.")
            engine.runAndWait()
            return  # 
        elif command:
        
            if "baby" in command or ("home" in command or "there" in command):
                engine.say("Did you just speak? If yes, speak again. If no, never mind.")
                engine.runAndWait()
            else:
                engine.say(" ")
                engine.runAndWait()


# Main function
def main():
    while True:
        command = listen_for_command("Say 'start recording video' to begin")
        if command == "start recording video":
            video_path = record_video()
            if video_path:
                process_decision = confirm_processing()  # Use new function for retries
                if process_decision is True:
                    objects = process_video(video_path)
                    unique_objects = {obj['object'] for obj in objects}
                    engine.say(f"In this room, I can see: {', '.join(unique_objects)}.")
                    engine.runAndWait()
                    interaction_loop(objects)
                elif process_decision is False:
                    engine.say("You can record another video or stop the interaction.")
                    engine.runAndWait()
                # If process_decision is None, it transitions to listening for the next command
            else:
                engine.say("Video recording failed. Please try again.")
                engine.runAndWait()
        elif command == "take a rest":
            engine.say("Taking a rest. Goodbye!")
            engine.runAndWait()
            break
        elif command in ["stop recording", "stop record"]:
            engine.say("Stopping the interaction. Listening for your next command.")
            engine.runAndWait()
            listen_for_next_command()
        else:
            engine.say("I didn't recognize that command. Please try again.")
            engine.runAndWait()

if __name__ == "__main__":
    main()
