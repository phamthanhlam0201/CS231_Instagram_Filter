import cv2
import mylib as lb

cap = cv2.VideoCapture(0)
filters = lb.load_filter()
i=0
f=False
while True: 
    ret, frame = cap.read()
    if not ret: break
    if cv2.waitKey(1) & 0xFF == ord('q'): break

    # frame = load_frame()
    points = lb.getLandmarks(frame)

    if not points or (len(points) != 75): continue

    for filter in filters:
        if filter['animated'] == False:
            frame = lb.apply_filter(frame,filter['data'],points)
        else:
            if lb.mouth_is_open(points) or f:
                frame = lb.apply_filter(frame,filter['datas'][i],points)
                f=True
                i += 1
                if i+1==len(filter['datas']): 
                    f=False
                    i=0

    cv2.imshow("Face Filter", frame)

cv2.destroyAllWindows()