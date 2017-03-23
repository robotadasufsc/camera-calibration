import argparse
import cv2 as cv
import numpy
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video",
                    help="Video file (-v video.webm) or camera (-v /dev/videoX).", default='video.webm')
parser.add_argument("-s", "--square",
                    help="Distance between squares in chessboard.", type=int, default=10)
parser.add_argument("-t", "--threshold",
                    help="Threshold error between chessboards, a lower value can take more processing time.", type=int, default=1)
parser.add_argument("-o", "--output",
                    help="Output file.", default='calib.yaml')
args = parser.parse_args()

if args.video:
    print('Source:', args.video)

if args.square:
    print('Distance between squares:', args.square)

if args.threshold:
    print('Threshold:', args.threshold)

if args.output:
    print('Output file:', args.output)

# Get video device or file
cap = cv.VideoCapture(args.video)

# Create pattern
pattern_size = (9, 6)
pattern_points = numpy.zeros((numpy.prod(pattern_size), 3), numpy.float32)
pattern_points[:, :2] = numpy.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= args.square

# Create calibration points
obj_points = []
img_points = []
target_perspective = []
for i in range(pattern_size[0]):
    for j in range(pattern_size[1]):
        target_perspective.append([50 * i, 50 * j])

matrix_vector = []

height = 0
width = 0

key = 0

success, frame = cap.read()
if success:
    height, width = frame.shape[:2]

while cap.isOpened():
    # Capture frame-by-frame
    success, frame = cap.read()

    print("%.2f %%" % float(100 * cap.get(cv.CAP_PROP_POS_FRAMES) /
                            cap.get(cv.CAP_PROP_FRAME_COUNT)))
    if not success:
        break

    if key == ord('q'):
        break
    else:
        found, corners = cv.findChessboardCorners(frame, pattern_size)

        if found:
            source_perspective = corners
            # Get homograph matrix
            matrix, _ = cv.findHomography(numpy.float32(
                source_perspective), numpy.float32(target_perspective))
            matrixb = matrix
            test = 0
            # Compare matrix with the
            for vmatrix in matrix_vector:
                for i in range(3):
                    for j in range(3):
                        # TODO Add cost function
                        matrixb[i][j] = numpy.sqrt(
                            abs(pow(matrix[i][j], 2) - pow(vmatrix[i][j], 2)))
                if(numpy.sum(matrixb) > args.threshold):
                    test = test + 1

            if(test == len(matrix_vector)):
                print('New good frame !', len(img_points))
                term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
                cv.drawChessboardCorners(frame, pattern_size, corners, found)
                img_points.append(corners.reshape(-1, 2))
                obj_points.append(pattern_points)
            matrix_vector.append(matrix)

    # Display the resulting frame
    cv.imshow('frame', frame)
    key = cv.waitKey(10)

# Close everything
cap.release()
cv.destroyAllWindows()

print("Best frames (%d / %d)" % (len(img_points), len(matrix_vector)))
print('Calculating camera calibration..')

# Calibrate
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(
    obj_points,
    img_points,
    (width, height),
    None,
    None)

calibration = {'rms': rms, 'camera_matrix': camera_matrix.tolist(),
    'dist_coefs': dist_coefs.tolist()}
print(calibration)
with open(args.output, 'w') as fw:
    print('Writing..', args.output)
    yaml.dump(calibration, fw)
