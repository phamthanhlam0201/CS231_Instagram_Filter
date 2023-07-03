import mediapipe as mp
import cv2
import math
import numpy as np
import csv
import imageio

filters = [{'path': "filters/dog-ears.png",
            'anno_path': "filters/dog-ears_annotations.csv",
            'animated': False, 'has_alpha': True},
            {'path': "filters/dog-nose.png",
            'anno_path': "filters/dog-nose_annotations.csv",
            'animated': False, 'has_alpha': True},
            {'path': "filters/dog-tongue.gif",
            'anno_path': "filters/dog-tongue_annotations.csv",
            'animated': True, 'has_alpha': True},
            ]

def similarityTransform(inPoints, outPoints):
  s60 = math.sin(60*math.pi/180)
  c60 = math.cos(60*math.pi/180)

  inPts = np.copy(inPoints).tolist()
  outPts = np.copy(outPoints).tolist()

  # The third point is calculated so that the three points make an equilateral triangle
  xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
  yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]

  inPts.append([np.int32(xin), np.int32(yin)])

  xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
  yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]

  outPts.append([np.int32(xout), np.int32(yout)])

  # Now we can use estimateRigidTransform for calculating the similarity transform.
  tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
  return tform[0]

def getLandmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]

    height, width = img.shape[:-1]

    with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return 0

        for face_landmarks in results.multi_face_landmarks:
            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2))

            for idx,value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y

            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')

            relevant_keypnts = []

            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return 0

def load_filter_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    b, g, r, alpha = cv2.split(img)
    img = cv2.merge((b, g, r))
    return img, alpha

def load_filter_gif(gif_path):
    loaded_gif = imageio.mimread(gif_path,memtest=False)
    gif=[]
    alpha=[]
    for frame in loaded_gif:
        r,g,b,a = cv2.split(frame)
        frame = cv2.merge((b,g,r))
        gif.append(frame)
        alpha.append(a)
    return gif, alpha

def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[int(row[0])] = (x, y)
            except ValueError:
                continue
        return points

def load_filter():
    multi_filter_runtime = []
    for filter in filters:
        temp_dict={}
        if filter['animated']:
            temp_dict['flt'], temp_dict['flt_a'] = load_filter_gif(filter['path'])
            temp_dict['animated'] = True
        else:
            temp_dict['flt'], temp_dict['flt_a'] = load_filter_img(filter['path'])
            temp_dict['animated'] = False
        temp_dict['points'] = load_landmarks(filter['anno_path'])
        
        multi_filter_runtime.append(temp_dict)
    return multi_filter_runtime

def apply_filter(frame, filter, points2):
    flt = filter['flt']
    points = filter['points']
    flt_alpha = filter['flt_a']
    dst_points = [points2[list(points.keys())[0]], points2[list(points.keys())[1]]]
    tform = similarityTransform(list(points.values()), dst_points)
    # Apply similarity transform to input image
    trans_flt = cv2.warpAffine(flt, tform, (frame.shape[1], frame.shape[0]))
    trans_alpha = cv2.warpAffine(flt_alpha, tform, (frame.shape[1], frame.shape[0]))
    mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

    # Blur the mask before blending
    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
    mask2 = (255.0, 255.0, 255.0) - mask1

    # Perform alpha blending of the two images
    temp1 = np.multiply(trans_flt, (mask1 * (1.0 / 255)))
    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
    output = temp1 + temp2

    frame = output = np.uint8(output)
    return frame

filters = load_filter()

frame = cv2.imread('353454151_1335018280729405_1196610630301947698_n.jpg',cv2.IMREAD_COLOR)
frame = cv2.resize(frame,(frame.shape[1]//4,frame.shape[0]//4))
points2 = getLandmarks(frame)
# print(points2[66])
# print(points2[62])
for filter in filters:
    if not filter['animated']:
        frame = apply_filter(frame, filter, points2)

cv2.imshow("Face Filter", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output.jpg", frame)