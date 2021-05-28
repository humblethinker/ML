import cv2
import numpy as np

#Colours Mapping for both functuons :
# Red : High risk
# Yellow : Low risk
# Green : Safe/No risk

# We will draw circles in the rectangular Bird's view corresponding to the transformed co-ordinates
# to the Bird view, the colour of of the circle corresponds to the risk level
def plot_top_view(frame, distances_mat, bottom_points, scale_w, scale_h, risk_count):
    h = frame.shape[0]
    w = frame.shape[1]

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (200, 200, 200)

    blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
    blank_image[:] = white
    warped_pts = []
    r = []
    g = []
    y = []
    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 0:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                r.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                r.append(distances_mat[i][1])

            blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)), (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1]* scale_h)), red, 2)
            
    for i in range(len(distances_mat)):
                
        if distances_mat[i][2] == 1:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                y.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                y.append(distances_mat[i][1])
        
            blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)), (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1]* scale_h)), yellow, 2)
            
    for i in range(len(distances_mat)):
        
        if distances_mat[i][2] == 2:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                g.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                g.append(distances_mat[i][1])
    
    for i in bottom_points:
        blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, green, 10)
    for i in y:
        blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, yellow, 10)
    for i in r:
        blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, red, 10)
        
    #pad = np.full((100,blank_image.shape[1],3), [110, 110, 100], dtype=np.uint8)
    #cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    #cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #blank_image = np.vstack((blank_image,pad))   
        
    return blank_image

# Each detected human will be contained in a box whose colour corresponds to risk level

def frame_view(frame, distances_mat, boxes, risk_count):
    
    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    
    for i in range(len(boxes)):

        x,y,w,h = boxes[i][:]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),green,2)
                           
    for i in range(len(distances_mat)):

        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]
        
        if closeness == 1:
            x,y,w,h = per1[:]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),yellow,2)
                
            x1,y1,w1,h1 = per2[:]
            frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),yellow,2)
            
    for i in range(len(distances_mat)):

        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]
        
        if closeness == 0:
            x,y,w,h = per1[:]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),red,2)
                
            x1,y1,w1,h1 = per2[:]
            frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),red,2)
            
    pad = np.full((140,frame.shape[1],3), [110, 110, 100], dtype=np.uint8)
    cv2.putText(pad, "Bounding box shows the level of risk to the person.", (50, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 0), 2)
    cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    frame = np.vstack((frame,pad))
            
    return frame

