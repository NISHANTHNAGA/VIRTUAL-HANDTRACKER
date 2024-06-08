from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")


col1, col2 = st.columns([2,1])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW=st.image([])

with col2:
    st.title("Answer")
    output_text_area=st.subheader("")

genai.configure(api_key="AIzaSyBXPdDIcj_v3FGDGf2M7-g4RK5udzUpCDU")
model = genai.GenerativeModel('gemini-1.5-flash')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def gethandinfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        
        hand1 = hands[0]  
        lmList = hand1["lmList"] 

        fingers = detector.fingersUp(hand1)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 1, 0, 0]:
        current_pos = tuple(map(int, lmList[8][0:2]))
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, (0, 0, 255), 10)
        prev_pos = current_pos
    
    elif fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(img)
        
    return current_pos, canvas

def sendtoai(model,canvas,fingers):
    if fingers ==[0,0,0,0,1]:
        pil_image= Image.fromarray(canvas)
        response = model.generate_content(["guess the image",pil_image])
        return response.text

prev_pos = None
canvas = None
image_combined=None
output_text=None

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = gethandinfo(img)
    if info:
        fingers,lmList=info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text=sendtoai(model,canvas,fingers)


    image_combined = cv2.addWeighted(img, 0.5, canvas,0.5, 0)
    FRAME_WINDOW.image(image_combined,channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    #cv2.imshow("Image", img)
    #cv2.imshow("Canvas", canvas)
    # cv2.imshow("Image Combined", image_combined)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()