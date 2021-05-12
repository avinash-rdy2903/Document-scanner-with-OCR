import numpy as np
import cv2
from scipy.ndimage import interpolation as inter
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
def get_sorted_contours_bounding_box(cnts,method='left-to-right'):
	reverse = False
	i = 0
	if method == "right-to-left" or method == "bottom-to-top":
		reverse =True
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

	return cnts, boundingBoxes
def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score
def skew_correction(img):
	wd, ht = img.shape[0],img.shape[1]
	pix = np.array(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), np.uint8)

	bin_img = 1 - (pix / 255.0).astype('int8')
	cv2.imwrite('binary.png',bin_img)

	delta = 1
	limit = 5
	angles = np.arange(-limit, limit+delta, delta)
	best_angle=-9999999
	best_score=-9999999
	for angle in angles:
		hist, score = find_score(bin_img, angle)
		if(best_score<score):
			best_score = score
			best_angle = angle
	print('Best angle: {}'.format(best_angle))
	# correct skew
	data = inter.rotate(img, best_angle, reshape=False, order=0)
	# img = cv2.cvtColor(data,cv2.COLOR_GRAY2BGR)
	data = cv2.GaussianBlur(data,(7,7),0)
	return data

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	transformation_matrix = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, transformation_matrix, (maxWidth, maxHeight))

	
	return warped