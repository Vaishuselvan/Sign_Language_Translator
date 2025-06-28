**Real-Time Sign Language Translator**
A GUI-based real-time American Sign Language (ASL) translator that uses computer vision and machine learning to interpret ASL hand gestures into spoken and written English. Powered by MediaPipe for hand tracking, Random Forest for classification, Tkinter for the user interface, and Text-to-Speech (TTS) for voice output.

**Why This Project?**
Communication barriers between hearing-impaired individuals and others can significantly impact inclusivity. This project aims to:

Enable real-time translation of ASL into text and speech

Provide an affordable and accessible solution using open-source technologies

Leverage machine learning and computer vision for practical societal benefit

**Tech Stack**
Technology	Purpose
MediaPipe	Real-time hand tracking and landmark detection
Tkinter	GUI interface for interaction
Random Forest Classifier	Machine Learning model to classify ASL gestures
Text-to-Speech (TTS)	Converts translated text to voice
OpenCV	Captures real-time webcam feed for gesture recognition
Python	Core programming language

**Model Performance**
Accuracy: ~95% on test data

Dataset: Custom ASL gesture images/videos

Preprocessing: Normalization and landmark feature extraction from MediaPipe

**Key Features**
 Real-time ASL gesture detection and translation

 Displays the detected gesture as text

 Converts translated text to speech output

 Easy-to-use Tkinter-based GUI

 Live webcam input for gesture detection
