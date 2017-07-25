from __future__ import print_function
import speech_recognition
import json
import utils.features
#import models.predict_emphasis

r = speech_recognition.Recognizer()
with speech_recognition.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

IBM_USERNAME = "bcb25809-0da2-47a5-b8b0-a3e57cfdabd9"  # IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
IBM_PASSWORD = "x1ByOkEPLGkJ"  # IBM Speech to Text passwords are mixed-case alphanumeric strings
try:
    mono_json = r.recognize_ibm(audio, username=IBM_USERNAME, password=IBM_PASSWORD, language="zh-CN", show_all = True)
    mono = json.loads(mono_json)
    print(mono)
except speech_recognition.UnknownValueError:
    print("IBM Speech to Text could not understand audio")
except speech_recognition.RequestError as e:
    print("Could not request results from IBM Speech to Text service; {0}".format(e))
