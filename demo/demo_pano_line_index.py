import VpEstimation
import cv2

canny = False

# canny = True

demo = "output_pano_line_index_"
batch = "2"

if canny:
    panoImage = cv2.imread("../dataset-good/dataset-" + str(21) + ".jpg", cv2.IMREAD_COLOR)
    print(panoImage.shape)
    panoLineIndex = VpEstimation.DrawPanoLineWithIndex(panoImage)
    cv2.imwrite("../output/" + demo + "0_canny.jpg", panoLineIndex)
else:
    for i in range(1, 30):
        panoImage = cv2.imread("../dataset-good/dataset-" + str(i) + ".jpg", cv2.IMREAD_COLOR)
        print(panoImage.shape, "----------")
        panoLineIndex = VpEstimation.DrawPanoLineWithIndex(panoImage)
        cv2.imwrite("../output/" + demo + "batch_" + batch + "_" + str(i) + ".jpg", panoLineIndex)
