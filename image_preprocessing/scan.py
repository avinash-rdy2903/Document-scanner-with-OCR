import numpy as np
import cv2
from skimage.filters import threshold_local
if(__name__ == "__main__"):
	from helper import four_point_transform,resize
else:
	from .helper import four_point_transform,resize
class FourPointException(Exception):
	def __init__(self):
		self.value='Cannot find a document to crop-in. \nPlease provide a valid picture of a document containig all four edges of it or a high quality Image :('
	def info(self):
		return repr(self.value)

def edge_detection(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blur, 50, 200)

	# print("STEP 1: Edge Detection")
	# cv2.imshow("Image", image)
	# cv2.imshow("Edged", edged)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return edged
def warper(orig,screenCnt,ratio,get_rgb=True):
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
	if(get_rgb):
		return warped
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	T = threshold_local(warped, 11, offset = 10, method = "gaussian")
	warped = (warped > T).astype("uint8") * 255

	# show the original and scanned images
	# print("STEP 3: Apply perspective transform")
	# cv2.imshow("Original", resize(img, height = 650))
	# cv2.imshow("Scanned", resize(warped, height = 650))
	# cv2.waitKey(0)
	return warped
def document_warper(image,get_rgb=True):
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = resize(image, height = 500)

	edged = edge_detection(image)

	cnts,hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
	screenCnt = None
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)
		# cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
		# cv2.imshow("Outline", image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
	
		if len(approx) == 4:
			screenCnt = approx
			break
	# print("STEP 2: Find contours of paper")
	# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
	# cv2.imshow("Outline", image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	try:
		warped = warper(orig, screenCnt,ratio,get_rgb)
	except AttributeError:
		raise FourPointException()
	return warped
