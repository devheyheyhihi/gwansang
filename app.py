import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from analysis import analyze_face
# from analysis import analyze_face # 얼굴 분석 기능은 나중에 추가합니다.
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 허용되는 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Haar cascade 로드
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!! ERROR: Haar cascade 파일 로드 실패: {face_cascade_path}")
    print("!! 'opencv-python' 패키지가 올바르게 설치되었는지,")
    print("!! 또는 해당 경로에 파일이 존재하는지 확인하세요.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    face_cascade = None
else:
    print(f"Haar cascade 파일 로드 성공: {face_cascade_path}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if face_cascade is None:
        return jsonify({'success': False, 'error': '서버 설정 오류: 얼굴 검출 모델을 로드하지 못했습니다.'})

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '파일이 업로드되지 않았습니다.'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '파일이 선택되지 않았습니다.'})
    
    if file and allowed_file(file.filename):
        try:
            # 파일을 메모리에서 직접 처리
            image_data = file.read()
            
            # 관상 분석 수행
            result = analyze_face(image_data)
            
            return jsonify({
                'success': True,
                'result': result
            })
            
        except Exception as e:
            print(f"분석 중 오류 발생: {str(e)}")
            return jsonify({'success': False, 'error': '이미지 분석 중 오류가 발생했습니다.'})
    
    return jsonify({'success': False, 'error': '지원하지 않는 파일 형식입니다.'})

@app.route('/result')
def result():
    data_param = request.args.get('data')
    if data_param:
        try:
            import json
            data = json.loads(data_param)
            return render_template('result.html', result=data.get('result'))
        except:
            pass
    
    return redirect(url_for('index'))

def analyze_face(image_data):
    """이미지 데이터에서 얼굴을 분석하여 관상 결과를 반환"""
    try:
        # 이미지 데이터를 numpy 배열로 변환
        from PIL import Image
        import io
        
        # 바이트 데이터를 PIL 이미지로 변환
        image = Image.open(io.BytesIO(image_data))
        
        # RGB로 변환 (RGBA나 다른 형식일 수 있으므로)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # numpy 배열로 변환
        img_array = np.array(image)
        
        # OpenCV 형식으로 변환 (BGR)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 얼굴 검출
        if face_cascade is None:
            raise RuntimeError("Face cascade is not loaded.")
            
        faces = face_cascade.detectMultiScale(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return {
                'face_detected': False,
                'message': '얼굴을 찾을 수 없습니다. 정면을 바라보는 선명한 사진을 업로드해주세요.'
            }
        
        # 가장 큰 얼굴 선택
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # 얼굴 영역 분석
        face_analysis = analyze_facial_features(img_bgr, x, y, w, h)
        
        # 관상학적 해석 생성
        fortune_reading = generate_fortune_reading(face_analysis)
        
        return {
            'face_detected': True,
            'analysis': face_analysis,
            'fortune': fortune_reading,
            'face_region': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        }
        
    except Exception as e:
        print(f"얼굴 분석 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'face_detected': False,
            'message': f'이미지 분석 중 오류가 발생했습니다: {str(e)}'
        }

def analyze_facial_features(img, x, y, w, h):
    """얼굴 특징을 분석하여 관상학적 데이터를 추출"""
    face_roi = img[y:y+h, x:x+w]
    
    # 얼굴 비율 분석
    face_ratio = w / h  # 가로/세로 비율
    
    # 얼굴 형태 분류
    if face_ratio > 0.85:
        face_shape = "둥근형"
    elif face_ratio < 0.75:
        face_shape = "긴형"
    else:
        face_shape = "타원형"
    
    # 얼굴 크기 분류
    face_area = w * h
    if face_area > 20000:
        face_size = "큰편"
    elif face_area < 10000:
        face_size = "작은편"
    else:
        face_size = "보통"
    
    # 이마 분석 (얼굴 상단 1/3)
    forehead_height = h // 3
    forehead_roi = face_roi[0:forehead_height, :]
    
    # 눈 영역 추정 (얼굴 중앙 1/3)
    eye_region_start = h // 3
    eye_region_end = 2 * h // 3
    eye_roi = face_roi[eye_region_start:eye_region_end, :]
    
    # 입 영역 추정 (얼굴 하단 1/3)
    mouth_region_start = 2 * h // 3
    mouth_roi = face_roi[mouth_region_start:, :]
    
    return {
        'face_shape': face_shape,
        'face_size': face_size,
        'face_ratio': round(face_ratio, 2),
        'forehead_proportion': round(forehead_height / h, 2),
        'eye_region_width': int(w),
        'mouth_region_height': int(h - mouth_region_start)
    }

def generate_fortune_reading(analysis):
    """얼굴 분석 결과를 바탕으로 관상학적 해석 생성"""
    
    # 얼굴형에 따른 성격 분석
    face_shape_meanings = {
        "둥근형": {
            "personality": "온화하고 친근한 성격으로 사람들과 잘 어울립니다. 포용력이 뛰어나고 협조적입니다.",
            "fortune": "인복이 많아 주변 사람들의 도움을 받기 쉽습니다. 안정적인 삶을 추구하며 점진적 발전이 예상됩니다.",
            "career": "서비스업, 상담, 교육 분야에서 재능을 발휘할 것입니다.",
            "health": "소화기 계통을 주의하시고, 규칙적인 운동으로 건강을 유지하세요."
        },
        "긴형": {
            "personality": "신중하고 사려깊은 성격입니다. 계획성이 뛰어나고 목표 지향적입니다.",
            "fortune": "늦게 피는 꽃으로, 중년 이후 크게 성공할 가능성이 높습니다. 꾸준한 노력이 결실을 맺을 것입니다.",
            "career": "연구, 기술, 전문직 분야에서 두각을 나타낼 것입니다.",
            "health": "스트레스 관리가 중요하며, 충분한 휴식을 취하시기 바랍니다."
        },
        "타원형": {
            "personality": "균형잡힌 성격으로 리더십이 뛰어납니다. 적응력이 좋고 다재다능합니다.",
            "fortune": "전반적으로 균형잡힌 운세를 가지고 있습니다. 노력한 만큼 성과를 거둘 것입니다.",
            "career": "관리직, 기획, 창업 등 다양한 분야에서 성공 가능성이 높습니다.",
            "health": "전반적으로 건강하나, 과로를 피하고 균형잡힌 생활을 유지하세요."
        }
    }
    
    face_shape = analysis['face_shape']
    base_reading = face_shape_meanings.get(face_shape, face_shape_meanings["타원형"])
    
    # 얼굴 크기에 따른 추가 해석
    size_modifier = ""
    if analysis['face_size'] == "큰편":
        size_modifier = " 큰 얼굴은 포용력과 리더십을 나타내며, 사회적으로 영향력을 발휘할 가능성이 높습니다."
    elif analysis['face_size'] == "작은편":
        size_modifier = " 작은 얼굴은 섬세함과 예술적 감각을 나타내며, 창조적 분야에서 재능을 발휘할 것입니다."
    
    # 이마 비율에 따른 지혜 운
    forehead_reading = ""
    if analysis['forehead_proportion'] > 0.35:
        forehead_reading = " 넓은 이마는 지혜와 통찰력을 상징하며, 학문이나 기획 분야에서 뛰어난 능력을 보일 것입니다."
    elif analysis['forehead_proportion'] < 0.3:
        forehead_reading = " 실용적이고 현실적인 사고를 가지고 있어, 실무 능력이 뛰어납니다."
    
    return {
        "overall": base_reading["personality"] + size_modifier,
        "fortune": base_reading["fortune"],
        "career": base_reading["career"],
        "health": base_reading["health"],
        "wisdom": forehead_reading if forehead_reading else "균형잡힌 사고력을 가지고 있어 다양한 분야에 적응할 수 있습니다.",
        "lucky_elements": get_lucky_elements(face_shape),
        "advice": get_life_advice(analysis)
    }

def get_lucky_elements(face_shape):
    """얼굴형에 따른 행운의 요소들"""
    elements = {
        "둥근형": {
            "color": "따뜻한 색상 (주황, 노랑)",
            "direction": "남쪽",
            "number": "2, 7",
            "stone": "호박, 황수정"
        },
        "긴형": {
            "color": "차분한 색상 (파랑, 보라)",
            "direction": "북쪽",
            "number": "1, 6",
            "stone": "사파이어, 자수정"
        },
        "타원형": {
            "color": "균형잡힌 색상 (녹색, 베이지)",
            "direction": "동쪽",
            "number": "3, 8",
            "stone": "에메랄드, 옥"
        }
    }
    return elements.get(face_shape, elements["타원형"])

def get_life_advice(analysis):
    """개인 맞춤 인생 조언"""
    advice = []
    
    if analysis['face_shape'] == "둥근형":
        advice.append("타인과의 관계를 소중히 하되, 자신의 의견도 당당히 표현하세요.")
        advice.append("안정을 추구하되, 가끔은 새로운 도전도 필요합니다.")
    elif analysis['face_shape'] == "긴형":
        advice.append("신중함은 장점이지만, 때로는 과감한 결정도 필요합니다.")
        advice.append("완벽을 추구하되, 완벽하지 않아도 괜찮다는 마음가짐을 가지세요.")
    else:
        advice.append("균형감각을 살려 다양한 기회를 잡으세요.")
        advice.append("리더십을 발휘하되, 겸손함도 잊지 마세요.")
    
    advice.append("건강한 생활습관을 유지하며, 정기적인 자기계발을 추천합니다.")
    advice.append("긍정적인 마음가짐으로 매일을 시작하세요.")
    
    return advice

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
