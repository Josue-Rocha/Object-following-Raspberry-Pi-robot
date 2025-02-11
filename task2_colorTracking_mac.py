# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2024
# ________________________________________________________________
# Adam Czajka, Andrey Kuehlkamp, September 2017 - 2024

# Here are your tasks:
#
# Task 2a:
# - Select one object that you want to track and set the RGB
#   channels to the selected ranges (found by colorSelection.py).
# - Check if HSV color space works better. Can you ignore one or two
#   channels when working in HSV color space? Why?
#	HSV probably worked a little bit better. You can try to ignore the value and/or saturation channels by placing a 0 in their place for both the upper and lower bounds. You can ignore channels because the HSV color spaces seperates the color, intensity, and purity. If you try to track multiple objects of the same color (before modifying the code) then the program will be forced to pick only one each loop.
# - Try to track candies of different colors (blue, yellow, green).
# 
# Task 2b:
# - Adapt your code to track multiple objects of *the same* color simultaneously, 
#   and show them as separate objects in the camera stream.
#
# Task 2c:
# - Adapt your code to track multiple objects of *different* colors simultaneously,
#   and show them as separate objects in the camera stream. Make your code elegant 
#   and requiring minimum changes when the number of different objects to be detected increases.
#
# Task for students attending 60000-level course:
# - Choose another color space (e.g., LAB or YCrCb), modify colorSelection.py, select color ranges 
#   and after some experimentation say which color space was best (RGB, HSV or the additional one you selected).
#   Try to explain the reasons why the selected color space performed best. 

import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while (True):
	retval, img = cam.read()

	res_scale = 0.5 # rescale the input image if it's too large
	img = cv2.resize(img, (0,0), fx = res_scale, fy = res_scale)


    	#######################################################
    	# Use colorSelection.py to find good color ranges for your object(s):

    	# Define multiple color ranges for object detection
	color_ranges = [
    		(np.array([0, 20, 110]), np.array([25, 70, 200])),
		(np.array([35, 90, 25]), np.array([70, 135, 65]))
    	]

	for lower, upper in color_ranges:
		objmask = cv2.inRange(img, lower, upper)

	    	# You may use this for debugging
		cv2.imshow("Binary image", objmask)

	    	# Resulting binary image may have large number of small objects
	    	# You may check different morphological operations to remove these unnecessary
	    	# elements. You may need to check your ROI defined in step 1 to
	    	# determine how many pixels your object may have.
		kernel = np.ones((5,5), np.uint8)
		objmask = cv2.morphologyEx(objmask, cv2.MORPH_CLOSE, kernel=kernel)
		objmask = cv2.morphologyEx(objmask, cv2.MORPH_DILATE, kernel=kernel)
		cv2.imshow("Image after morphological operations", objmask)

	    	# find connected components
		cc = cv2.connectedComponents(objmask)
		ccimg = cc[1].astype(np.uint8)

	    	# Find contours of these objects
		contours, hierarchy = cv2.findContours(ccimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

	    	# We are using [-2:] to select the last two return values from the above function to make the code work with
	    	# both opencv3 and opencv4. This is because opencv3 provides 3 return values but opencv4 discards the first.

	    	# You may display the countour points if you want:
	    	# cv2.drawContours(img, contours, -1, (0,255,0), 3)

	    	# Ignore bounding boxes smaller than "minObjectSize"
		minObjectSize = 30;


	    	#######################################################
	    	# TIP: think if the "if" statement
	    	# can be replaced with a "for" loop
		for c in contours:

			# use the biggest object to draw a rectangle
		    	# c = max(contours, key = cv2.contourArea)
			x, y, w, h = cv2.boundingRect(c)

		    	#######################################################
		    	# TIP: you want to get bounding boxes
		    	# of ALL contours (not only the first one)
		    	#######################################################

		    	# Do not show very small object
			if w > minObjectSize or h > minObjectSize:
				cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
				cv2.putText(img,            # image
				"Object Detected",        # text
				(x, y-10),                  # start position
				cv2.FONT_HERSHEY_SIMPLEX,   # font
				0.7,                        # size
				(0, 255, 0),                # BGR color
				1,                          # thickness
				cv2.LINE_AA)                # type of line

	cv2.imshow("Live WebCam", img)

	action = cv2.waitKey(1)
	if action==27:
        	break
    
cam.release()
cv2.destroyAllWindows()
(cv1p1) josuerocha@Josues-MacBook-Pro ~ % 
