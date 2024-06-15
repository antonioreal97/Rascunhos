import cv2
from cvzone.HandTrackingModule import HandDetector
from pynput.keyboard import Key, Controller, Listener

video = cv2.VideoCapture(0)

video.set(3,1280)
video.set(4,720)

kb = Controller()

detector = HandDetector(detectionCon=0.8)
estadoAtual = [0,0,0,0,0]

setaDir = cv2.imread('seta dir.PNG')
setaEsq = cv2.imread('seta esq.PNG')

#def on_press(key):
#    if key == Key('b'):
#        print("Breakpoint reached!")
#        listener.stop()  # Stop the listener
#        cv2.destroyAllWindows()  # Close all OpenCV windows
#        exit()  # Exit the program

#listener = Listener(on_press=on_press)
#listener.start()

while True:
    _,img = video.read()
    hands,img = detector.findHands(img)

    if hands:
        estado = detector.fingersUp(hands[0])
    
        print(estado)

        if estado!=estadoAtual and estado == [0,0,0,0,1]:
            print('passar slide')
            kb.press(Key.right)
            kb.release(Key.right)

        if estado!=estadoAtual and estado == [1,0,0,0,0]:
            print('voltar slide')
            kb.press(Key.left)
            kb.release(Key.left)

        if estado == estadoAtual and estado == [0, 0, 0, 0, 1]:
            img[50:216, 984:1230] = setaDir
        if estado == estadoAtual and estado == [1, 0, 0, 0, 0]:
            img[50:216, 50:296] = setaEsq

        estadoAtual = estado
    cv2.imshow('img',cv2.resize(img,(640,420)))
    cv2.waitKey(1)