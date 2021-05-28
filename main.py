import cv2
import numpy as np
import time
import argparse

# own modules
import utills, plot

confid = 0.5
thresh = 0.5
mouse_pts = []


# We take 8 points as input and store their raw co-ordinates, first 4 points help us determine the region in which we would to do the surveillance, the 4 points will be quadrilateral's 4 vertices in anti-clockwise order.
# Then 5-6 points will determine the ratio of 180 cm to pixels in horizontal direction
# Then 6-7 points will determine the ratio of 180 cm to pixels in vertical direction
# Point 8 determines that all inputs have been taken

def store_mouse_clicks(event, x, y, flags, param):

    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)
            
        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)
        
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        

# Main function of the project
def COVID_protocol_monitor(vid_path, net, output_dir, output_vid, ln1):
    
    count = 0
    vs = cv2.VideoCapture(vid_path)    

    # Video's dimensions and frame per second
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    
    # Set scale for birds eye view
    # Bird's eye view will only show ROI
    scale_w, scale_h = utills.change_scale(width, height)

    points = []
    global image
    
    while True:

        (grabbed, frame) = vs.read()

        if not grabbed:
            print('here')
            break
            
        (H, W) = frame.shape[:2]
        
        # For, first frame we pause it to take the Mouse point inputs
        if count == 0:
            while True:
                image = frame
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 8:
                    cv2.destroyWindow("image")
                    break
               
            points = mouse_pts      
                 
        # Using the first 4 points to determine and set the Bird's eye correspondence to the given view
        # by using homography through OpenCV's getPerspectiveTransform()
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)

        # Getting the Bird's eye view coordinates from raw co-ordinates using the previously set PerspectiveTransform() object
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]
        
        # Using Pythagorus theorem to get the distance between point 5 and 6 and distance between 6 and 7
        # which are the pixel distance in Bird's view corresponding to 180cm in real life
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)
    
    ####################################################################################
    
        # Inputting the frame image into the pre-trained YOLO v3 model
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln1)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []   
    
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # Checking if humans are present or not
                if classID == 0:

                    if confidence > confid:

                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                    
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x,y,w,h = boxes[i]
                
        if len(boxes1) == 0:
            count = count + 1
            continue

        # We will treat a person as two dimensional coordinates in which the height part is set zero and
        # only x and y coordinates from origin are of importance and origin is defined by the fourth click by the user
        person_points = utills.birds_view_coordinates(boxes1, prespective_transform)
        
        # Generating a distance matrix 
        distances_mat, bxs_mat = utills.closeness_calculator(boxes1, person_points, distance_w, distance_h)
        risk_count = utills.count_risk(distances_mat)
    
        frame1 = np.copy(frame)
        
        # Plot circles for humans with colours corresponding to their risk level  
        bird_image = plot.plot_top_view(frame, distances_mat, person_points, scale_w, scale_h, risk_count)
        img = plot.frame_view(frame1, bxs_mat, boxes1, risk_count)
        
        # Write output to the respective locations 
        # For Bird eye window, it is OpenCV window
        # For Frame view, it will be output folder
        if count != 0:
    
            cv2.imshow('Bird Eye View', bird_image)
            cv2.imwrite(output_dir+"frame%d.jpg" % count, img)
            cv2.imwrite(output_dir+"plot_top_view/frame%d.jpg" % count, bird_image)
    
        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    vs.release()
    cv2.destroyAllWindows() 
        

if __name__== "__main__":

    # Receives arguements specified by user
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-v', '--video_path', action='store', dest='video_path', default='C:/Users/humbl/Documents/Social/data/example.mp4' ,
                    help='Path for input video')
                    
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='C:/Users/humbl/Documents/Social/output/' ,
                    help='Path for Output images')

    parser.add_argument('-O', '--output_vid', action='store', dest='output_vid', default='C:/Users/humbl/Documents/Social/output_vid/' ,
                    help='Path for Output videos')

    parser.add_argument('-m', '--model', action='store', dest='model', default='C:/Users/humbl/Documents/Social/models/',
                    help='Path for models directory')
                    
    parser.add_argument('-u', '--uop', action='store', dest='uop', default='NO',
                    help='Use open pose or not (YES/NO)')
                    
    values = parser.parse_args()
    
    model_path = values.model
    if model_path[len(model_path) - 1] != '/':
        model_path = model_path + '/'
        
    output_dir = values.output_dir
    if output_dir[len(output_dir) - 1] != '/':
        output_dir = output_dir + '/'
    
    output_vid = values.output_vid
    if output_vid[len(output_vid) - 1] != '/':
        output_vid = output_vid + '/'


    # Load YOLO v3 model
    
    weightsPath = model_path + "yolov3.weights"
    configPath = model_path + "yolov3.cfg"

    net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net_yl.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]

    # Set mouse back

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", store_mouse_clicks)
    np.random.seed(42)
    
    COVID_protocol_monitor(values.video_path, net_yl, output_dir, output_vid, ln1)



