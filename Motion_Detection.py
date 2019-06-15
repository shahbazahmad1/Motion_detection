import cv2, pandas
from datetime import datetime
import imutils


first_frame = None
video = cv2.VideoCapture(0)

statusList = [-1, -1]
times = []
df = pandas.DataFrame(columns=["Start", "End"])

fps = 30
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('Feed.avi',
                              cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'),
                              fps, size)


while True:
    check, frame = video.read()
    status = -1
    text = 'Unoccupied'
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

    if first_frame is None:
        first_frame = grayImg
    else:
        pass


    deltaFrame = cv2.absdiff(first_frame, grayImg)
    threshFrame = cv2.threshold(deltaFrame, 30, 255, cv2.THRESH_BINARY)[1]
    threshFrame = cv2.dilate(threshFrame, None, iterations=2)


    (cnts, _) = cv2.findContours(threshFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        text = 'Occupied'

        continue

    cv2.putText(frame, '[Room Status]: %s' % (text),
                (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.50, (255, 255, 255), 1)

    cv2.putText(frame, datetime.now().strftime('%A %d %B %Y %I:%M:%S%p'),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.50, (255, 255, 255), 1)
    videoWriter.write(frame)
    statusList.append(status)


    if (statusList[-1] * statusList[-2]) == -1:

     times.append(datetime.now())


     cv2.imshow("Capturing", grayImg)
     cv2.imshow("DeltaFrame", deltaFrame)
     cv2.imshow("Threshold Frame", threshFrame)
     print(status)



    cv2.imshow("Colour Frame", frame)


    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:

            times.append(datetime.now())
        break

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i + 1]}, ignore_index=True)


df.to_csv("TimeList.csv")
cv2.destroyAllWindows()
video.release()