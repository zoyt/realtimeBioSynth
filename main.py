# originally built on techVidvan Hand Gesture Recognizer code, but heavily modified


# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
from pyo import *

# also "pip install wrapt==1.12.1" ref by tensorflow
# also "pip install msvc-runtime" referenced by mp

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(1)

# Initialize pyo for audio output
s = Server().boot()
s.start()


def angle_3p_3d(a, b, c):  # calculates angle between three points in 3d space
    v1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    v2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])

    v1mag = np.sqrt([v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]])
    v1norm = np.array([v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag])

    v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
    v2norm = np.array([v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag])
    res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
    angle_rad = np.arccos(res)

    return math.degrees(angle_rad)


def percentClosed(sum, offset):  # helper function for finger closed calculation, poss not used
    return int(((sum / 3)-offset) * 100 / offset)


# Play demo tune, fairly random
t = CosTable([(0,0), (100,1), (500,.3), (8191,0)])
beat = Beat(time=.125, taps=16, w1=[90,80], w2=50, w3=35, poly=7).play()
trmid = TrigXnoiseMidi(beat, dist=12, mrange=(60, 96))
trhz = Snap(trmid, choice=[0,2,3,5,7,8,10], scale=1)
tr2 = TrigEnv(beat, table=t, dur=beat['dur'], mul=beat['amp'])
a = Sine(freq=trhz, mul=tr2*0.3).out()


# main function
while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    className = ''

    THUMB_CMC_ANGLE, THUMB_MPC_ANGLE, THUMB_IP_ANGLE = 0, 0, 0
    INDEX_MCP_ANGLE, INDEX_PIP_ANGLE, INDEX_DIP_ANGLE = 0, 0, 0
    MIDDLE_MPC_ANGLE, MIDDLE_PIP_ANGLE, MIDDLE_DIP_ANGLE = 0, 0, 0
    RING_MPC_ANGLE, RING_PIP_ANGLE, RING_DIP_ANGLE = 0, 0, 0
    PINKY_MPC_ANGLE, PINKY_PIP_ANGLE, PINKY_DIP_ANGLE = 0, 0, 0

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        coords = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                lmz = int(lm.z * -10000)

                landmarks.append([lmx, lmy])

                coords.append([lmx, lmy, lmz])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            print(coords)
            # calculate 3d angles for all angle sets
            THUMB_CMC_ANGLE = angle_3p_3d(coords[0], coords[1], coords[2])
            THUMB_MPC_ANGLE = angle_3p_3d(coords[1], coords[2], coords[3])
            THUMB_IP_ANGLE = angle_3p_3d(coords[2], coords[3], coords[4])

            INDEX_MCP_ANGLE = angle_3p_3d(coords[0], coords[5], coords[6])
            INDEX_PIP_ANGLE = angle_3p_3d(coords[5], coords[6], coords[7])
            INDEX_DIP_ANGLE = angle_3p_3d(coords[6], coords[7], coords[8])

            MIDDLE_MPC_ANGLE = angle_3p_3d(coords[0], coords[9], coords[10])
            MIDDLE_PIP_ANGLE = angle_3p_3d(coords[9], coords[10], coords[11])
            MIDDLE_DIP_ANGLE = angle_3p_3d(coords[10], coords[11], coords[12])

            RING_MPC_ANGLE = angle_3p_3d(coords[0], coords[13], coords[14])
            RING_PIP_ANGLE = angle_3p_3d(coords[13], coords[14], coords[15])
            RING_DIP_ANGLE = angle_3p_3d(coords[14], coords[15], coords[16])

            PINKY_MPC_ANGLE = angle_3p_3d(coords[0], coords[17], coords[18])
            PINKY_PIP_ANGLE = angle_3p_3d(coords[17], coords[18], coords[19])
            PINKY_DIP_ANGLE = angle_3p_3d(coords[18], coords[19], coords[20])

    # Create array of joint angles, reshape for readability
    reshape = np.array([THUMB_CMC_ANGLE, THUMB_MPC_ANGLE, THUMB_IP_ANGLE,
                        INDEX_MCP_ANGLE, INDEX_PIP_ANGLE, INDEX_DIP_ANGLE,
                        MIDDLE_MPC_ANGLE, MIDDLE_PIP_ANGLE, MIDDLE_DIP_ANGLE,
                        RING_MPC_ANGLE, RING_PIP_ANGLE, RING_DIP_ANGLE,
                        PINKY_MPC_ANGLE, PINKY_PIP_ANGLE, PINKY_DIP_ANGLE
                        ]).reshape(5, 3)
    reshape = reshape.astype(int)
    reshape = np.rot90(reshape)
    reshape = np.flipud(reshape)

    # Assign calculation offsets
    thumb_offset, index_offset, mid_offset, ring_offset, pinky_offset = 85, 120, 100, 120, 120


    # Calculate percetage extended each finger
    thumb_sum, index_sum, mid_sum, ring_sum, pinky_sum = np.sum(reshape, axis=0)

    # precise_thumb_avj = percentClosed(thumb_sum, 90)  # angle sum, finger offset
    # precise_index_avj = percentClosed(index_sum, 100)  # angle sum, finger offset
    # precise_mid_avj = percentClosed(mid_sum, 120)  # angle sum, finger offset
    # precise_ring_avj = percentClosed(ring_sum, 120)  # angle sum, finger offset
    # precise_pinky_avj = percentClosed(pinky_sum, 120)  # angle sum, finger offset
    # averages = np.array((precise_thumb_avj, precise_index_avj, precise_mid_avj, precise_ring_avj, precise_pinky_avj))

    coarse_thumb = int(thumb_sum/3)
    coarse_index = int(index_sum / 3)
    coarse_mid = int(mid_sum / 3)
    coarse_ring = int(ring_sum / 3)
    coarse_pinky = int(pinky_sum / 3)

    averages = np.array([coarse_thumb, coarse_index, coarse_mid, coarse_ring, coarse_pinky])

    reshape = np.vstack((averages, reshape))

    # average (180-150)/180 = %/100

    # Display all the information
    text = str(reshape[:][:])
    y0, dy = 50, 15
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(frame, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # # change attributes of the music based on the % extended values
    # beat.taps = int(np.interp(thumb_avj,[0, 100], [5, 25]))  # smoothly interpolate across values acceptable as beats
    # beat.time = float(np.interp(mid_avj,[0, 100], [0.1, 0.75]))
    # a.freq = int(np.interp(index_avj,[0, 100], [500, 2000]))  # smoothly interpolate across values acceptable as frequencies

    # change attributes of the music based on the % extended values
    beat.taps = int(np.interp(coarse_thumb, [160, 180], [5, 20]))  # smoothly interpolate across values acceptable as beats
    beat.time = float(np.interp(coarse_index, [100, 180], [0.1, 0.2]))
    a.freq = int(np.interp(coarse_mid, [100, 180], [200, 700]))  # smoothly interpolate across values acceptable as frequencies


    # display the frame with mapped on skeleton
    cv2.imshow("Output", frame)

    # if q is pressed, stop everything
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()

s.stop()

