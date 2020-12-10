OpenPose : Caffe와 OpenCV를 기반으로 구성된 손, 얼굴 포함 몸의 움직임을 추적해주는 API

1. 아래의 깃허브에서 오픈 포즈 다운로드<br>
https://github.com/CMU-Perceptual-Computing-Lab/openpose

2. getModels로 모델을 다운로드
..\openpose-master\models\pose\coco 폴더의 COCO모델 사용
- pose_deploy_linevec.prototxt
- pose_iter_440000.caffemodel

<p>COCO모델에서 제공하는 POINT
COCO Output Format Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, LAnkle – 13, Right Eye – 14, Left Eye – 15, Right Ear – 16, Left Ear – 17, Background – 18
우리 프로젝트에서 상체의 움직임중 끄덕임등 목과 머리의 움직임을 파악하기 위해서 Nose, Neck, Right Eye, Left Eye, Right Ear, Left Ear 사용.</p>
<br><br>

<h3>reference</h3>
https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/ <br>
https://github.com/natanielruiz/deep-head-pose



