from typing import List
from flask import Flask, request, Response, redirect, jsonify
import jsonpickle
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# import the necessary packages
from skimage.filters import threshold_local
import imutils
from scipy import ndimage
import math


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Elastic Beanstalk looks for an 'application' that is callable by default
application = Flask(__name__)

@application.route('/')
def hello_world():
    return 'Hello, World!'

@application.route('/histogram/image', methods=['POST'])
def histogram_file():
	# check if the post request has the file part
	if 'file' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message' : 'No file selected for uploading'})
		resp.status_code = 400
		return resp
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join('', filename))
		image = cv2.imread(filename)
		img_data = test(image)
		hist, bin_edges = np.histogram(img_data, range=(0, max(img_data)), bins=10)
		x_coord = []
		i = 0
		for edge in bin_edges:
			if i == len(bin_edges) - 1:
				break
			mid_point = (bin_edges[i + 1] + edge) / 2
			x_coord.append(mid_point)
			i += 1
		os.remove(filename)
		hist = hist.tolist()
		resp = jsonify({'x_axis' : x_coord, 'y_axis' : hist})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
		resp.status_code = 400
		return resp

@application.route('/analyze/image', methods=['POST'])
def analyze_file():
	# check if the post request has the file part
	if 'file' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message' : 'No file selected for uploading'})
		resp.status_code = 400
		return resp
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join('', filename))
		image = cv2.imread(filename)
		img_data = test(image)
		os.remove(filename)
		resp = jsonify({'data' : str(img_data)})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
		resp.status_code = 400
		return resp

def test(image):
	# r = request
	# convert string of image data to uint8
	# nparr = np.fromstring(r.data, np.uint8)
	# decode image
	# image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	# load the image and compute the ratio of the old height
	# to the new height, clone it, and resize it
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)
	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (15, 15), 0)
	edged = cv2.Canny(gray, 75, 200)
	
	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = False)[:5]
	# just use the largest contour area
	arc = cv2.arcLength(cnts[0], True)
	maxCnts = cv2.approxPolyDP(cnts[0], 0.03 * arc, True)

	minX = arc # arc is the area, this makes sure we're always over shooting
	minY = arc
	maxX = 0
	maxY = 0

	for p in maxCnts:
		x = p[0][0]
		if x > maxX:
			maxX = x
		if x < minX:
			minX = x
		
		y = p[0][1]
		if y > maxY: 
			maxY = y
		if y < minY:
			minY = y

	# for c in cnts:
	# 	peri = cv2.arcLength(c, True)
	# 	approx = cv2.approxPolyDP(c, 0.03 * peri, True)
	# 	if len(approx) == 4:
	# 		screenCnt = approx
	# 		break

	# apply the four point transform to obtain a top-down
	# view of the original image
	points = np.array([[minX, minY], [minX, maxY], [maxX, minY], [maxX, maxY]])

	warped = four_point_transform(orig, points.reshape(4, 2) * ratio)

	dimensions = warped.shape
	min_x = 0
	min_y = 0
	max_x = dimensions[1]
	max_y = dimensions[0]

	src = warped

	gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5,5), 0)
	maxValue = 255
	adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C#cv2.ADAPTIVE_THRESH_MEAN_C #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
	thresholdType = cv2.THRESH_BINARY#cv2.THRESH_BINARY #cv2.THRESH_BINARY_INV
	blockSize = 5 #odd number like 3,5,7,9,11
	C = -3 # constant to be subtracted
	im_thresholded = cv2.adaptiveThreshold(gray, maxValue, adaptiveMethod, thresholdType, blockSize, C) 

	labelarray, particle_count = ndimage.measurements.label(im_thresholded)
	kernel = np.ones((5,5),np.uint8)
	#kernal = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
	dilation = cv2.dilate(im_thresholded,kernel,iterations = 1)
	#opening = cv2.morphologyEx(im_thresholded, cv2.MORPH_CLOSE, kernel)

	delta_x_px = max_x - min_x
	delta_y_px = max_y - min_y

	ratio = delta_y_px / delta_x_px
	# Can we shore this up to get the ratio as we expect?

	delta_x_mm = 215.9
	delta_y_mm = 279.4

	ratio = delta_y_mm / delta_x_mm

	pixels_per_mm_x = delta_x_px / delta_x_mm
	pixels_per_mm_y = delta_y_px / delta_y_mm

	pixels_per_mm_avg = (pixels_per_mm_x + pixels_per_mm_y) / 2

	contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
	res = np.array(src)
	areas_in_mm = []
	for c in contours:
		x, y, widthPixel, heightPixel = cv2.boundingRect(c)
		# Make sure the contours are far enough in from the edges
		if(float(x) - min_x > 200.0) and (float(y) - min_y > 200.0) and (max_x - float(x) > 200.0) and (max_y - float(y)> 200.0):
			if(cv2.contourArea(c) > 30.0):         
				area = cv2.contourArea(c)
				area_w_px = math.sqrt(area)
				area_h_px = math.sqrt(area)
				area_w_mm = area_w_px / pixels_per_mm_avg
				area_h_mm = area_h_px / pixels_per_mm_avg
				area_mm = area_w_mm * area_h_mm
				# filter out the fines! validate this!!
				if area_mm > 1:
					areas_in_mm.append(area_mm)
				center_x = round(x + (widthPixel / 2))
				center_y = round(y + (heightPixel / 2))
				x1 = round(center_x - (area_w_px / 2)) 
				y1 = round(center_y - (area_h_px / 2))
				x2 = round(center_x + (area_w_px / 2))
				y2 = round(center_y + (area_h_px / 2))
	return areas_in_mm

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect
	
def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


# Run the application
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production application.
    application.debug = True
    application.run(host="0.0.0.0")