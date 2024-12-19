FEATURES: 
The system is designed to deliver a rich set of functionalities. It uses the YOLO model for real-time object detection and integrates MiDaS for depth estimation, allowing precise measurement of distances. Voice-based interaction enables users to initiate and control the system effortlessly through speech recognition. Recorded video inputs are analyzed to extract information about the environment, and OpenAI GPT is leveraged to generate detailed navigation instructions tailored to the detected objects and user queries. Additionally, the system merges duplicate object detections from multiple frames for increased accuracy and reliability.

SYSTEM ARCHITECTURE: 
The system operates in multiple interconnected stages. First, it captures video using a webcam and processes the frames for object detection and depth estimation. YOLO identifies objects in the video frames while MiDaS estimates the depth of objects to determine their distances. Using a speech recognition module, the system listens to user commands and provides feedback or instructions via text-to-speech. Detected objects and user queries are processed through OpenAI GPT, which generates step-by-step navigation instructions based on the spatial relationship of objects in the environment.

REQUIREMENTS: 
This project relies on several Python libraries, including opencv-python, torch, pyttsx3, openai, speechrecognition, and others. A webcam is required for video input, and a microphone is needed for voice commands. For optimal performance, a system with a CUDA-enabled GPU is recommended to accelerate the computational workload. Before running the project, ensure all dependencies are installed using pip.

SETUP AND USAGE: 
To use the system, start by cloning the repository and setting up your OpenAI API key in the code. Afterward, run the script to initialize the system. Users can start video recording by saying "start recording video" and stop it with the "stop recording" command. Once a video is recorded, the system processes the data to identify objects and calculate distances. Users can then ask questions about the detected objects, and the system will respond with navigation instructions.

HOW IT WORKS: 
The workflow begins with the user commanding the system to start video recording. The system records the environment and listens for the "stop recording" command. Once the recording is complete, the video is processed for object detection and depth estimation. The detected objects and their spatial information are used to generate navigation instructions in response to user queries. The system continues to remain active, listening for additional commands, making it ideal for dynamic and interactive environments.
