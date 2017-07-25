from __future__ import print_function
import speech_recognition
import json
import sys
sys.path.append("..")
import utils.features
sys.path.append("..")
#import models.predict_emphasis

r = speech_recognition.Recognizer()
with speech_recognition.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

IBM_USERNAME = "bcb25809-0da2-47a5-b8b0-a3e57cfdabd9"  # IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
IBM_PASSWORD = "x1ByOkEPLGkJ"  # IBM Speech to Text passwords are mixed-case alphanumeric strings
try:
    mono = r.recognize_ibm(audio, username=IBM_USERNAME, password=IBM_PASSWORD, language="zh-CN", show_all = True)
    mono = mono["results"][0]["alternatives"][0]["timestamps"]
    print(mono)
    text = []
    for item in mono:
        text.append(item[0])
    print(''.join(text))
    raw_data = audio.get_raw_data()

    models.predict_emphsis.ext_feature(raw_data)

except speech_recognition.UnknownValueError:
    print("IBM Speech to Text could not understand audio")
except speech_recognition.RequestError as e:
    print("Could not request results from IBM Speech to Text service; {0}".format(e))
