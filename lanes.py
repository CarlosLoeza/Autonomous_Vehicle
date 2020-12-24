import cv2
import numpy as np
import matplotlib.pyplot as plt # Allow us to recognize lines on road

# changes our image to grayscale
# param:
#   - image or video of road
def canny(image):
    # change the grayscale of our copied image, lane_image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # canny
    canny = cv2.Canny(blur, 50, 150)
    return canny


# Determines the coordinates we will use
# parameters:
#   - image: image or video of road
#   - line_parameters: left or right line
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0] # bottom
    y2 = int(y1*(3/5)) # reaches 3/5 above the bottom
    x1 = int((y1-intercept) / slope) # x coordinate: attain by x = (y-b) / m
    x2 = int((y2-intercept) / slope)
    # return an array with make_coordinates
    return np.array([x1, y1, x2, y2])


# This will give us an average slope from all of the lines in that specific side (Left or Right line)
# parameters:
#   - image: image or video or road
#   - lines: lines detected from reading the car lane
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        # get slope and intercept of line
        slope = parameters[0]
        intercept = parameters[1]
        # if slope negative: left lane
        if slope < 0:
            left_fit.append((slope, intercept))
        # else: it is a right lane
        else:
            right_fit.append((slope, intercept))
    # average: compute the average slope of all the left and right lines and
    #          make one of each
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    # Get the coordinates for the average slope for left and right lane
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    # return as array
    return np.array([left_line, right_line])


# This function will help us display the lines/lanes in blue
# parameters:
#   - image: image or video of road
#   - lines: lanes
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            #line(image, start, end, color, line thickness)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image


# Tell our program where to look.
# In this case we get a triangle shaped to read the lanes
# parameters:
#   - image: image or video of road
def region_of_interest(image):
    height = image.shape[0]
    # create an array with coordinates of our triangle
    polygons = np.array([[(200, height), (1100, height), (550,250)]])
    # mask it with zeros
    mask = np.zeros_like(image)
    # get outline of triangle
    cv2.fillPoly(mask, polygons, 255)
    # masked_image: only show the region of interest between canny and masked image(black and white)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image




# _____________ MAIN _____________________________________

# image processing
cap = cv2.VideoCapture("test2.mp4")

while(cap.isOpened()):
    # decodes video frame
    _, frame = cap.read()
    canny_image = canny(frame)
    # cropped_image: all black image with white lanes
    cropped_image = region_of_interest(canny_image)
    # lines: highlights the lanes from cropped_image using HoughLinesP
    # HoughLinesP(image, pixel, degree_precison, threshold, placeholder array, length of line, gap )
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # averaged_lines: create an average lane based on our data
    averaged_lines = average_slope_intercept(frame, lines)
    # line_image: image of lines highlight in blue
    # param:
    #   - lane_image: photo of lines aka lanes
    #   - lines: lines found using HoughLinesP
    line_image = display_lines(frame, averaged_lines)
    # combine_photos: combines both photos and intensifies the colors
    # we want our lines to be darker
    combine_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    # .imshow(Title we want to give image, image passed)
    cv2.imshow("result", combine_image)
    # breakout of loop if q is pressed
    if cv2.waitKey(1) == ord('q'):
        break;

# Close current window
# The window normally shows our image/video with lane detection
cap.release()
cv2.destroyAllWindows()
