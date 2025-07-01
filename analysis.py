import cv2
import numpy as np
from math import hypot
import os

def analyze_face(image_path):
    """
    OpenCV를 사용하여 얼굴을 분석하고 결과를 반환하는 함수
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "error": "이미지를 불러올 수 없습니다. 파일 경로를 확인해주세요."}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # OpenCV의 Haar Cascade를 사용한 얼굴 검출
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return {"success": False, "error": "사진에서 얼굴을 찾을 수 없습니다. 더 선명하거나 정면을 바라보는 사진을 이용해주세요."}
        
        # 첫 번째 검출된 얼굴만 분석
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # 얼굴 영역에서 눈 검출
        eyes = eye_cascade.detectMultiScale(face_roi)
        
        # 부위별 분석 (OpenCV 기반으로 단순화)
        face_analysis = analyze_face_features(faces[0], eyes, img.shape)
        
        analysis_result = {
            "success": True,
            "features": {
                "얼굴형": face_analysis['face_shape'],
                "눈": face_analysis['eyes'],
                "전체적인 인상": face_analysis['overall_impression'],
                "운세": face_analysis['fortune']
            },
            "overall": face_analysis['summary']
        }
        
        return analysis_result
        
    except Exception as e:
        return {"success": False, "error": f"이미지 처리 중 오류 발생: {e}"}

def analyze_face_features(face_rect, eyes, img_shape):
    """
    검출된 얼굴과 눈의 정보를 바탕으로 관상 분석
    """
    (x, y, w, h) = face_rect
    img_height, img_width = img_shape[:2]
    
    # 얼굴 비율 분석
    face_ratio = w / h
    face_size_ratio = (w * h) / (img_width * img_height)
    
    # 얼굴형 분석
    if face_ratio > 0.85:
        face_shape = "둥근 얼굴형으로 온화하고 친근한 인상을 주며, 대인관계가 원만하고 복이 많은 상입니다."
        face_score = 8
    elif face_ratio < 0.75:
        face_shape = "긴 얼굴형으로 지적이고 신중한 성격을 나타내며, 학문이나 전문분야에서 성공할 가능성이 높습니다."
        face_score = 7
    else:
        face_shape = "균형 잡힌 얼굴형으로 안정적이고 조화로운 성격을 가지고 있으며, 리더십이 뛰어납니다."
        face_score = 9
    
    # 눈 분석
    if len(eyes) >= 2:
        eyes_analysis = "두 눈이 또렷하게 보이며 눈빛이 살아있어 총명하고 의지가 강한 상입니다. 목표 달성 능력이 뛰어나고 재물운이 좋습니다."
        eyes_score = 8
    elif len(eyes) == 1:
        eyes_analysis = "한쪽 눈이 강조되어 보이며 개성이 강하고 독립적인 성향을 나타냅니다. 창의적인 분야에서 두각을 나타낼 수 있습니다."
        eyes_score = 6
    else:
        eyes_analysis = "눈의 형태가 부드럽고 온화하여 인자하고 포용력이 있는 성격입니다. 주변 사람들에게 신뢰받는 상입니다."
        eyes_score = 7
    
    # 전체적인 인상
    if face_size_ratio > 0.15:
        impression = "얼굴이 크고 당당하여 리더의 기질이 있으며, 사회적으로 인정받고 명예를 얻을 가능성이 높습니다."
        impression_score = 8
    else:
        impression = "단정하고 깔끔한 인상으로 섬세하고 신중한 성격을 나타내며, 꾸준한 노력으로 성공을 이룰 수 있습니다."
        impression_score = 7
    
    # 종합 점수 계산
    total_score = (face_score + eyes_score + impression_score) / 3
    
    # 운세 분석
    if total_score >= 8:
        fortune = "대길(大吉) - 복과 덕을 갖춘 매우 귀한 상입니다. 재물운, 명예운, 건강운이 모두 좋으며 주변의 도움을 많이 받을 것입니다."
    elif total_score >= 7:
        fortune = "길(吉) - 전체적으로 좋은 운세를 가지고 있습니다. 꾸준한 노력과 긍정적인 마음가짐으로 원하는 바를 이룰 수 있습니다."
    elif total_score >= 6:
        fortune = "중길(中吉) - 평범하지만 안정적인 운세입니다. 성실함과 인내심으로 점진적인 발전을 이룰 수 있습니다."
    else:
        fortune = "소길(小吉) - 현재는 다소 아쉬운 부분이 있으나, 긍정적인 변화와 노력을 통해 운명을 개척해 나갈 수 있습니다."
    
    # 종합 요약
    if total_score >= 7.5:
        summary = "매우 좋은 관상을 가지고 계십니다. 타고난 복과 함께 노력하는 만큼 큰 성취를 이룰 수 있는 상입니다."
    elif total_score >= 6.5:
        summary = "균형 잡힌 좋은 관상입니다. 안정적인 삶과 함께 꾸준한 발전을 이룰 수 있는 상입니다."
    else:
        summary = "현재 상태에서도 충분히 좋은 점들이 많이 보입니다. 긍정적인 마음과 꾸준한 노력으로 더 나은 미래를 만들어갈 수 있습니다."
    
    return {
        'face_shape': face_shape,
        'eyes': eyes_analysis,
        'overall_impression': impression,
        'fortune': fortune,
        'summary': summary
    } 