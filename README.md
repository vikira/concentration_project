# AI 온라인 수업 조교 - "교수님 모르겠어요" 

-스타트22팀 졸업프로젝트입니다.

## 시연영상(유튜브 주소)


# 프로젝트 소개

 **교수님 모르겠어요**는 사회적 거리두기로 인해 실시되고 있는 실시간 온라인 수업에서 학생들의 수업에 대한 반응 및 이해도 파악을 돕는 AI 서비스입니다. 이 서비스는 시선 추적, 표정 인식, 행동 인식을 통해 학생들의 수업 이해도를 실시간으로 분석하여수업에 대한 실시간 피드백이 이루어질 수 있습니다. 교수자가 더욱 질 높은 수업을 제공할 수 있게 되어 교육자와 학생 모두의 수업에 대한 만족도를 상승시키는 것을 목표로 합니다.
 ## 1. 전제
 - 캠을 키고하는 수업하는 강의식 수업에 한정함
- 사용자가 수업 화면 하나만을 보고 있다고 가정함
- 참여자 모두의 이해도 통합을 목표로 하지만 1인 상황파악을 우선적으로 시작함
 ## 2. 모듈화
  <img width="481" alt="화면 캡처 2020-12-10 192338" src="https://user-images.githubusercontent.com/63234878/101759671-3ab59080-3b1d-11eb-8b00-f3dcfca10b66.png">

### 반응파악모듈
웹캠을 통한 영상에서 시선, 표정, 행동을 인식하여 정보를 추출. 수업 참여 판단 모듈과 이해도 모듈로 정보를 전달.
### 수업참여판단모듈
1. 수업 듣는 중 (1)
웹캠에 얼굴이 인식
시선이 모니터안에 인식
2. 수업 안듣는중 (0)
ex) 졸음 : 고개를 숙인 상태, 눈을 장시간 감고있는 상태
웹캠에 얼굴 or 시선이 인식되지 않음
->인식 안됨이 지속될 시 수업을 듣지 않고 있다고 판별
### 이해도모듈
#### 산출 시나리오
1. 이해도 증감은 수업을 듣고 있는 학생(수업 듣는 중(1))에 한해서만 반영
2. 5초 간격으로 이해도 리셋 후 재측정을 반복 
3. 측정시마다 이해도는 중립 상태(50%) 에서 시작
4. 고개 끄덕끄덕 거리는 것을 2회 이상하거나 화면을 바라보며 미소를 지으면 이해도 증가로 반영
5. 화면을 바라보며 2초 이내에 동공 좌표 네 번 이상 변화(동공지진), 하품, 찡그림이 감지 되면 이해도 감소로 반영
6. openCV로 인식한 슬라이드의 컨텐츠가 아닌 부분을 3초 이상 응시하고 있으면 이해도 감소로 반영
7. 손으로 이마나 머리를 짚음, 턱을 괴고 있음 등의 행동이 감지되면 이해도 감소로 반영
#### 분석결과 제공 시나리오
1. 1인 이해도를 통합하여 전체 이해도 계산
2. 수업이 진행되는 동안에는 시간별로 수업의 진도 상황을 기록
3. 미리 업로드한 교수의 수업자료에서 opencv로 각 슬라이드 인식
4. 각 슬라이드와 진도상황 기록한 것을 대조해, 학생들의 이해도가 낮은 부분을 따로 기록
5. 수업이 끝난 후, 이해도가 낮은 부분에 대해 시간과 슬라이드 기준으로 그래프로 그려서 교수자에게 제공

## 3. 기술
### 반응파악모듈 -시선인식
1)Gaze Tracking API: github.com/antoinelame/GazeTracking

사용모델
dlib shape_predictor_68_face_landmarks.dat

- 왼쪽 눈동자의 좌표
```
gaze.pupil_left_coords()
```
- 오른쪽 눈동자의 좌표
```
gaze.pupil_right_coords()
```
- 눈동자의 방향
```
gaze.is_left()
gaze.is_right()
gaze.is_center()
```
- 눈 감음
```
gaze.is_blinking()
```

2)콘텐츠 검출 API: github.com/nicewoong/pyTextGotcha
이미지 전처리를 위해 다음과 같은 다섯 단계를 거친다.
- gray scale 적용
```
def gray_scale(img)
```
- Morph Gradient, Morph Close 적용
```
def gradient(img_gray)
```
- Adaptive Threshold 적용 
```
image_threshold = cv2.adaptiveThreshold(copy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV)
```
- Long Line Removal 적용
```
int threshold; #선 추출 정확도
int minLength; #추출할 선의 길이
int lineGap; #이 픽셀 이내로 겹치는 선은 제외


lines = cv2.HoughLinesP(copy, 1, np.pi / 180, threshold, np.array([]), min_line_length, max_line_gap)
for line in lines:
	x1, y1, x2, y2 = line[0]  # end point of line
    	cv2.line(copy, (x1, y1), (x2, y2), (0, 0, 0), 2)
```
- Contour영역 잘라내기

### 반응파악모듈 -표정인식
1. OpenCV dlib -  68 face landmark로 찡그림, 하품, 졸림, 지겨움 등 표정 파악
ex) 눈과 눈썹 point 사이 거리 추출 하여 찡그림 인식

### 반응파악모듈 -행동인식
#### OpenPose
Caffe와 OpenCV를 기반으로 구성된 손, 얼굴 포함 몸의 움직임을 추적해주는 API <br>
https://github.com/CMU-Perceptual-Computing-Lab/openpose
<br>
- pose_deploy_linevec.prototxt
- pose_iter_440000.caffemodel
<br>
COCO모델에서 제공하는 POINT<br>
Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, LAnkle – 13, Right Eye – 14, Left Eye – 15, Right Ear – 16, Left Ear – 17, Background – 18 <br>

상체의 움직임중 끄덕임등 목과 머리의 움직임을 파악하기 위해서 Nose, Neck, Right Eye, Left Eye, Right Ear, Left Ear 사용. Pose pair중 필요한 상체 연결만 남김.<br>

	POSE_PAIRS = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[0,14],[0,15],[14,16],[15,17]]

<div>
  <img width="259" alt="화면 캡처 2020-12-10 205622" src="https://user-images.githubusercontent.com/63234878/101769370-35ab0e00-3b2a-11eb-85e2-72b065f7d5ee.png">
</div>
<br>
webcam으로 캡쳐한 이미지를 전처리

    inpBlob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0 / 255, (300,300), (0, 0, 0), swapRB=False, crop=False)

네트워크를 통과한 후 몸의 관절을 하이라이트한 confidence map에서 global maxima를 가진 point추출. 추출된 포인트중 piar를 이루는 점들을 연결.

    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
#### 끄덕임 검출<br>
Nose(0)과 Neck(1)이 같이 검출되었을 때 그 포인트 사이의 거리를 계산하고, 머리가 중립 상태일 때의 바운더리를 설정하여 그 이하, 이상으로 변화하면 움직임을 검출


## 4. reference
시선인식
https://wwww.medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6

표정인식 

행동인식
https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/ <br>


## 5. License 
## 프로젝트 진행 과정
- [x] 주제선정 및 내용구체화
- [x] 기술조사 및 코드리뷰
- [ ] 각 모듈 개발
- [ ] 이해도 모델 생성
- [ ] 모듈 통합
- [ ] UI 및 프로토타입 제작
- [ ] 서버 구축
