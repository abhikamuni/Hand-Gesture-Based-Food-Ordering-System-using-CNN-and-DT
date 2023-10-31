import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from sklearn.decomposition import PCA

def load_models():
    cnn_model = load_model('cnn_model.h5')
    decision_tree_model = joblib.load('decision_tree_model.pkl')
    return cnn_model, decision_tree_model

def initialize_camera(resolution=(640, 480)):
    cap = cv2.VideoCapture(0)
    cap.set(3, resolution[0])
    cap.set(4, resolution[1])
    return cap

def process_image(hand_img, hands_model):
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    output = hands.process(img_flip)
    hands.close()

    # Implement image processing and feature extraction
    if output.multi_hand_landmarks:
        data = output.multi_hand_landmarks[0]
        data = str(data)
        data = data.strip().split('\n')

        # Process and extract landmarks
        # ...

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])


        return img_flip, np.array(clean).flatten()
    else:
        return img_flip, np.zeros([1, 63], dtype=int)[0]

def main():
    cnn_model, decision_tree_model = load_models()
    cap = initialize_camera()

    imgBackground = cv2.imread("Resources/Background.png")

    # storing all the modes into a list
    modes_path = "Resources/Modes"
    image_mode_path = os.listdir(modes_path)
    image_mode = []

    for image in sorted(image_mode_path,key=lambda x:int(x.replace('.png', ''))):
        image_mode.append(cv2.imread(os.path.join(modes_path, image)))

    # storing all the Icons into a list
    icons_path = "Resources/Icons"
    icon_path_list = os.listdir(icons_path)
    image_icons = []

    for icons in sorted(icon_path_list,key=lambda x:int(x.replace('.png', ''))):
        image_icons.append(cv2.imread(os.path.join(icons_path,icons )))

    mode_type = 0  # for changing the mode

    #detector = HandDetector(detectionCon=0.8, maxHands=1)

    selection = 0
    counter = 0
    selection_speed = 36
    #mode_positions = [(1294,196),(1053,414),(1294,636)] # 0,1,2
    mode_positions = [(816,196),(1053,196),(1294,196),(816,414),(1053,414),(1294,414),(816,636),(1053,636),(1294,636)]

    counter_pause = 0
    order_complete = 0
    flag = 0
    store_selected_items = [0,0,0]

    if not cap.isOpened():
        print("Cannot open camera")
        exit()



    while True:
        ret, frame = cap.read()
        x_offset = 50
        y_offset = 139

        # Resize the webcam frame to match the region size
        frame = cv2.resize(frame, (640, 480))
    
        # overlaying the webcam feed to the background image
        # ration is frame width and height
        imgBackground[139:139+480, 50:50+640] = frame
        imgBackground[0:775, 704:1435] = image_mode[mode_type]
        #imgBackground[0:720, 847:1280] = image_mode[mode_type]
    
    

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        processed_image, cnn_feature_vector = process_image(frame, cnn_model)

        # Ensure the CNN feature vector has 63 features
        cnn_feature_vector_63 = np.pad(cnn_feature_vector, (0, 63 - len(cnn_feature_vector)), 'constant')

        # Predict using the CNN model
        y_pred_cnn = cnn_model.predict(cnn_feature_vector_63.reshape(1, 63))

        # Predict using the decision tree model
        y_pred_decision_tree = decision_tree_model.predict(cnn_feature_vector_63.reshape(1, 63))

        predicted_class = y_pred_decision_tree[0]

        # Mapping of class indices to gesture names (customize this based on your dataset)
        gesture_names = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5",6: "6", 7: "7", 8: "8", 9: "9", 10: "10"}

        gesture_label = gesture_names.get(predicted_class, "Unknown")

        if predicted_class and mode_type == 3:
            if predicted_class == 0:
                if order_complete > 0:
                    order_complete += 1
                    if order_complete >= 30:
                        flag = 1


        if predicted_class and counter_pause == 0 and mode_type < 4:
        

            #here the 1 represent index finger this condition will execute. when the user show index finger
            if predicted_class == 1:
                if selection != 1:
                    counter = 1
                selection = 1
            elif predicted_class == 2:
                if selection != 2:
                    counter = 1
                selection = 2
            elif predicted_class == 3:
                if selection != 3:
                    counter = 1
                selection = 3
            elif predicted_class == 4:
                if selection != 4:
                    counter = 1
                selection = 4
            elif predicted_class == 5:
                if selection != 5:
                    counter = 1
                selection = 5
            elif predicted_class == 6:
                if selection != 6:
                    counter = 1
                selection = 6
            elif predicted_class == 7:
                if selection != 7:
                    counter = 1
                selection = 7
            elif predicted_class == 8:
                if selection != 8:
                    counter = 1
                selection = 8
            elif predicted_class == 9:
                if selection != 9:
                    counter = 1
                selection = 9
            else:
                selection = 0
                counter = 0

            if counter > 0:
                counter += 1
                # print(counter)
                """
                    here giving the image path -> imgBackground
                    center to show mode_positions ((1136,196),(1000,384),(1136,581)) based on the image that i have, (0,1,2) indexes.
                    axes : (103,103)
                    angle : 0
                    start angle : 0
                    end angle : counter * seletionspeed
                    color : green in rbg(0,255,0)
                    thickness of the layer : 20
                """
                cv2.ellipse(imgBackground,mode_positions[selection-1],(90,90),0,0,counter*selection_speed,(0,255,0),8)

                if counter*selection_speed > 360:
                    store_selected_items[mode_type] = selection
                    if mode_type == 2 and selection == 2:
                        mode_type +=1 
                    mode_type += 1
                    counter = 0
                    selection = 0
                    counter_pause = 1

                if mode_type == 3:
                    order_complete = 1
                if mode_type == 4:
                    order_complete = 0

        if counter_pause > 0:
            counter_pause += 1
            if counter_pause >= 60:
                counter_pause = 0


        # Display the predicted gesture label on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (100, 200)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        imgBackground = cv2.putText( imgBackground,gesture_label, org, font, fontScale, color, thickness, cv2.LINE_AA)


        if store_selected_items[0] != 0:
            imgBackground[636:636 + 65,133:133 + 65] = image_icons[store_selected_items[0]-1]
        if store_selected_items[1] != 0:
            imgBackground[636:636 + 65,340:340 + 65] = image_icons[8+store_selected_items[1]]
        if store_selected_items[2] != 0:
            imgBackground[636:636 + 65,542:542 + 65] = image_icons[17+store_selected_items[2]]


        cv2.imshow('Food ordering system', imgBackground)

        if cv2.waitKey(1) == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
