import streamlit as st
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from keras.models import load_model
import mediapipe as mp
import time
st.set_page_config(
    page_title="나의 MBTI는?",
    page_icon="face_favicon.png",
    menu_items={
        'About': "나의 mbti는?\nThis is an cool app!"
    }
)
kakao_ad_code1 = """
 <ins class="kakao_ad_area" style="display:none;"
data-ad-unit = "DAN-1ygiN7CoxyoZKd4X"
data-ad-width = "250"
data-ad-height = "250"></ins>
<script type="text/javascript" src="//t1.daumcdn.net/kas/static/ba.min.js" async></script>
"""
kakao_ad_code2 = """
 <ins class="kakao_ad_area" style="display:none;"
data-ad-unit = "DAN-8C0tMmtzAzgVLIeO"
data-ad-width = "250"
data-ad-height = "250"></ins>
<script type="text/javascript" src="//t1.daumcdn.net/kas/static/ba.min.js" async></script>
"""
coupang_ad_code="""
<iframe src="https://ads-partners.coupang.com/widgets.html?id=718831&template=carousel&trackingCode=AF3660738&subId=&width=680&height=140&tsource=" width="680" height="140" frameborder="0" scrolling="no" referrerpolicy="unsafe-url"></iframe>
<style>margin: 0 auto;</style>
"""
@st.cache_resource
def get_model():
	model=load_model('keras_model.h5',compile=False)
	class_names=open('labels.txt','r').readlines()
	return model,class_names
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(0,255,0))
def main():
	model,class_names=get_model()
	st.title("_나의 MBTI는_?:cupid:")
    
    # 파일 업로드 섹션 디자인
	st.subheader('인공지능이 당신의 MBTI를 분석해줄거에요!:sunglasses:')
	st.write(':blue[얼굴 정면 사진을 업로드 해주세요! 사진은 저장되지 않습니다!]')
    # 파일 업로드 컴포넌트
	uploaded_file = st.file_uploader("PNG 또는 JPG 이미지를 업로드하세요.", type=["png", "jpg", "jpeg"])
	if uploaded_file is not None:
        # 이미지를 넘파이 배열로 변환
		image = Image.open(uploaded_file).convert('RGB')
		image = ImageOps.exif_transpose(image)
		img_np=np.array(image)
		try:
			with mp_face_detection.FaceDetection(
					model_selection=1, min_detection_confidence=0.5) as face_detection:
					results=face_detection.process(img_np)
					annotated_image = img_np.copy()
			with mp_face_mesh.FaceMesh(
					static_image_mode=True,
					max_num_faces=1,
					refine_landmarks=True,
					min_detection_confidence=0.5) as face_mesh:
				img = annotated_image
				# 작업 전에 BGR 이미지를 RGB로 변환합니다.
				results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
				annotated_image = img.copy()
				for face_landmarks in results.multi_face_landmarks:
					mp_drawing.draw_landmarks(
						image=annotated_image,
						landmark_list=face_landmarks,
						connections=mp_face_mesh.FACEMESH_TESSELATION,
						landmark_drawing_spec=drawing_spec,
						connection_drawing_spec=mp_drawing_styles
						.get_default_face_mesh_tesselation_style())
			st.image(annotated_image,caption='업로드한 이미지',use_column_width=True)
			data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
			size = (224, 224)
			image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
			image_array = np.asarray(image)
			normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
			data[0] = normalized_image_array
			prediction = model.predict(data)
			index = np.argmax(prediction)
			result=class_names[index][2:].strip()
			confidence_score = prediction[0][index]
			with st.spinner('AI가 당신의 MBTI를 분석중입니다...'):
				time.sleep(3)  # 예시로 3초 동안 로딩 중 표시 (실제 분석으로 대체 필요)
				st.success(f'MBTI분석을 완료했습니다!\n\n나의 MBTI는? {result} {round(confidence_score*100)}%')
			tab1,tab2,tab3,tab4=st.tabs(['특징','연애 스타일','추천직업','유명인'])
			if result=='INFP':
				with tab1:
					st.subheader('#열정적인 :blue[중재자형]')
					st.write('이타적인 성격을 가진 유형이며  타인의 감정을 세심하게 살피며 열정적입니다. 장점으로는 공감 능력과 감수성이 뛰어난 편이며 타인을 존중하고 배려한다는 점입니다. 단점으로는 행동력이 떨어지고 인간관계에서 상처를 많이 받는다는 점이 단점입니다.')
				with tab2:
					st.subheader('#이상적인 사랑을 꿈꾸는, :blue[연애의 로맨티스트]')
					st.write('언제나 로맨틱한 영화 같은 사랑을 꿈꾸고 있는 INFP. 연애를 시작함과 동시에 앞으로 행복할 일, 싸울 일, 우울할 일을 미리 상상하고 기뻐하며 걱정하는 스타일입니다. 사귀기 전에는 벽을 치지만, 사귄 후에는 언제 그랬냐는 듯이 애정표현을 적극적으로 합니다.')
					st.write('최고의 궁합: ISFP\n\n최악의 궁합: ESFP')
				with tab3:
					st.subheader('#열정적인 :blue[중재자]')
					st.write('예술가, 소설가, 시인, 음악가, 미술 치료사, 사회복지사, 작곡가, 사서')
				with tab4:
					st.subheader('#INFP :blue[연예인]')
					st.write('여자 유명인: 한효주, 노윤서, 한소희, 뉴진스 하니, 트와이스 모모')
					st.write('남자 유명인: 이준기, 박서준, BTS 뷔, 위너 송민호, 강하늘')
			elif result=='INFJ':
				with tab1:
					st.subheader('#선의의 :blue[옹호자형]')
					st.write('통찰력이 높고 섬세한 언어 표현을 하고 공동체 생활을 중요시 생각합니다. 대표적인 INFJ 특징으로는 낯가림이 심하고, 계획적이라는 것 입니다. INFJ 장점 키워드는 친절, 섬세함, 이타적, 신중함, 수용, 부지런함 이며 반면, 단점 키워드는 내성적이고 다소 예민하며, 현실성이 부족하다는 점입니다. 은근 고집이 세서 타인과 가치가 충돌하는 상황이 발생할 때,가끔 그 상황 자체를 회피하고자 한다는 점도 있습니다.')
				with tab2:
					st.subheader('#이해와 배려로 깊은 사랑을 꿈꾸는, :blue[연애의 비전가]')
					st.write('물리적 거리가 가까울수록 애정의 크기도 커집니다. 하지만 표현이 서툴러서 짝사랑과 속앓이를 가장 많이 하는 스타일이기도 합니다. 표정관리를 너무 잘해서 상대가 마음을 전혀 눈치 채지 못할 때도 많습니다. 하지만 연애를 시작하기만 하면 연애를 방해하는 모든 장애물을 다 부수며 상대에게 올인합니다.')
					st.write('최고의 궁합: ESFP\n\n최악의 궁합: ESTJ')
				with tab3:
					st.subheader('#선의의 :blue[옹호자]')
					st.write('직업상담사, 특수 교사, 노인 복지사, 아트 디렉터, 프리랜서 기획, 저널리스트, 상품기획 MD')
				with tab4:
					st.subheader('#INFJ :blue[연예인]')
					st.write('여자 유명인: 송지효, 송혜교, 설인아, 아이유, 태연')
					st.write('남자 유명인: 조인성, 서강준, 남주혁, 안보현, 태양')
			elif result=='INTP':
				with tab1:
					st.subheader('#논리적인 :blue[사색가형]')
					st.write('조용하고 과묵하면서 평범함을 거부하며 독창적이고 창의적인 사람입니다. 관심사를 연구, 탐구하는 일에 시간을 쏟는 편이고, 가까운 사람에게만 마음을 여는 타입입니다. 대표적인 INTP 특징으로는 자신만의 세계가 확고하다는 점입니다. 또한, 장점으로는 아이디어가 풍부하고, 영리하다는 점,그리고 효율적으로 문제 해결을 한다는 점이 있습니다. 반면, 개인주의에 반항적이고, 사회성이 부족하다는 단점이 있습니다.')
				with tab2:
					st.subheader('#합리적으로 사랑을 탐색하는, :blue[연애의 탐구자]')
					st.write('귀찮은 것을 싫어하고 선을 넘는 것도 싫어합니다. 한편으로는 솔직하다고 볼 수 있지만 또 다른 편으로는 무심하다고 느낄 수 있습니다. 기존의 연애방식을 거부하고 나만의 연애를 만들어가며 그렇기에 유니크하고 특별한 연애를 하는 경우가 많습니다. 상대의 외모나 조건보다 사고방식, 대화 스타일이 나와 잘 맞는지가 중요합니다.')
					st.write('최고의 궁합: ENFJ\n\n최악의 궁합: INFJ')
				with tab3:
					st.subheader('#논리적인 :blue[사색가]')
					st.write('경제학자, 심리학자, 경찰, 프로그래머, 천문학자, 비평가, 아트디렉터, 연구원')
				with tab4:
					st.subheader('#INTP :blue[연예인]')
					st.write('여자 유명인: 권나라, 초아, 신민아, 고준희, 르세라핌 사쿠라')
					st.write('남자 유명인: BTS 진, BTS 슈가, 송강, 김우빈, 안효섭')
			elif result=='INTJ':
				with tab1:
					st.subheader('#용의주도한 :blue[전략가형]')
					st.write('문제에 대해 의문을 던지고 규칙을 깨는 일에 두려움이 없으며, 시간단위로 계획을 짤 만큼 철저한 계획을 세우며 독립적인 성향이 있다. INTJ의 결단력 있고, 논리적인 것이 장점이다. 평소에는 조용한 편이지만, 전문적인 지식을 논할 때는 매우 적극적인 편이다. INTJ의 단점은 직설적이고 고집이 세다는 것입니다. “개인주의자"라고 불릴 만큼이나 이기적이며, 융통성이 없다는 점이다.')
				with tab2:
					st.subheader('#효율적인 :blue[사랑의 전략가], 연애도 논리로 접근')
					st.write('자신의 감정을 정말 잘 숨깁니다. 분석적이면서 냉철하다는 평가를 받으며 호불호가 강하고 주관이 뚜렷해서 가치관이 맞지 않으면 관심을 갖지 않습니다. 조용하고 깊은 사랑을 추구하는데, 연애 스타일 또한 이슈 없이 무난한 연애를 합니다. 사랑을 시작하기에는 오랜 시간이 걸릴 수 있지만 한번 시작하면 깊은 사랑을 하는 타입입니다.')
					st.write('최고의 궁합: ENFP\n\n최악의 궁합: INTJ')
				with tab3:
					st.subheader('#용의주도한 :blue[전략가]')
					st.write('분석가, 회계사, 인류학자, 파일럿, 경영 컨설턴트, 제약회사 연구원, 웹 개발자, 최고 재무 책임자')
				with tab4:
					st.subheader('#INTJ :blue[연예인]')
					st.write('여자 유명인: 안소희, ITZY 류진, 우주소녀 루다, 프로미스나인 박지원, 보아')
					st.write('남자 유명인: 강동원, 이수혁, 지드래곤, 에릭')
			elif result=='ISTJ':
				with tab1:
					st.subheader('#청렴경백한 :blue[논리주의자형]')
					st.write('사실과 근거에 집중하여 꼼꼼한 유형 매사 철저하게 행동하며 종사하는 곳에서 핵심적인 역할을 가지고 있습니다. 대표적인 ISTJ 특징은 매사에 계획적이고 신중하다는 것입니다. ISTJ 장점으로는 논리적이고 규칙을 잘 지키는 모범적인 면이 있습니다.반면, 지나친 원칙주의자이며, 낯가림이 심하고, 감정표현에 미숙하여 사교성이 부족하다는 점이 단점입니다.')
				with tab2:
					st.subheader('#연애도 철저한 계획대로, 믿을 수 있는 :blue[사랑의 선봉장]')
					st.write('꼼꼼하고 철두철미하며 감정표현도 서툴지만 내적 사랑이 가득한 스타일입니다. 연애도 나만의 원칙에 맞게 계획적으로 차근차근 진행하기 때문에 시작이 다소 느릴 수 있습니다. 그 때문에 주변에서 연애를 하는지도 모르는 경우가 있기도 합니다. 신중하다 보니 본인 감정에 대한 선을 잘 지키고, 이로 인해 연애 기간이 긴 편입니다.')
					st.write('최고의 궁합: ENFP\n\n최악의 궁합: INFP')
				with tab3:
					st.subheader('#청렴 결백한 :blue[논리주의자]')
					st.write('통계학자, 바이어, 기상학자, 법률 연구원, 보험 심사관, 형사, 감정 평가사, 세관 조사관')
				with tab4:
					st.subheader('#ISTJ :blue[유명인]')
					st.write('여자 유명인: 소녀시대 써니, 우주소녀 보나, 이보영, 여자친구 신비, IVE 가을')
					st.write('남자 유명인: 이창섭, 갓세븐 마크, 이석훈, 차태현')
			elif result=='ISTP':
				with tab1:
					st.subheader('#:blue[만능재주꾼형]')
					st.write('조용하지만, 관찰력이 뛰어난 성격의 유형이다. 만드는 것과 도구를 다루는 것에 관심이 많고 판단력이 매우 빠른 편 입니다. 대표적인 특징은 조용하다가도 자기주장에는 적극적이라는 것입니다. 주변에서 "태어난 김에 산다" 혹은 "마이웨이"라는 소리를 많이 듣는 편이라고 하지만, 낯가림이 심할 뿐, 가까운 사람들에게는 따뜻한 편이며, 말보다는 행동으로 보여주는 스타일입니다.')
				with tab2:
					st.subheader('#융통성 있고 즉흥적인, :blue[연애의 행동주의자]')
					st.write('자신만의 기준이 합리적이면서 명확합니다. 연애할 때도 저돌적인 스타일이 아닌 선을 지키면서 단계를 밟아가는 사랑을 추구합니다. 사생활을 중요시 여기며 혼자 있는 시간이 부족하다고 여길 경우 스트레스를 받습니다. 애정표현은 다소 소극적인 편이라 좋아하는 사람이 생겨도 티가 잘 안 납니다.')
					st.write('최고의 궁합: ESFJ\n\n최악의 궁합: ESFP')
				with tab3:
					st.subheader('#:blue[만능 재주꾼]')
					st.write('파일럿, 카레이서, 범죄학자, 사진 작가, 판매원, 운동선수, 항공기 정비사, 네트워크 관리자')
				with tab4:
					st.subheader('#ISTP :blue[유명인]')
					st.write('여자 유명인: 김연아, 고윤정, 트와이스 나연, 르세라핌 김채원, 안유진')
					st.write('남자 유명인: 덱스, 박명수, 주우재, 은지원, 김종민')
			elif result=='ISFJ':
				with tab1:
					st.subheader('#용감한 :blue[수호자형]')
					st.write('차분한 성격으로 사람들과 우호적인 관계를 맺는 유형 대표적인 ISFJ 특징으로는 섬세하고 성실하다는 점입니다. 자기 주관이나 신념이 있어도 겉으로 잘 드러내지 않고, 규칙/위계질서 등에 있어 엄격합니다. 매사에 차분하고 온화한 성격과, 성실하고 책임감 있는 태도를 가지고 있는 것이 장점이며 단점으로는 본인의 규칙에서 어긋나는 경우, 다소 예민해지고 판단력을 잃을 수 있다는 점입니다.')
				with tab2:
					st.subheader('#내 마음속에 담긴 섬세한 감정, :blue[사랑의 애정 가득한 수호자]')
					st.write('나의 노력을 존중해주고 진심으로 감사할 줄 아는 사람을 만나면 오랜 기간 연애를 합니다. 평화의 수호신이라고 이야기할 수 있을 정도로 어떤 일이 있더라도 잘 참으면서 맞춰줍니다. 누구보다 아낌 없이 사랑을 주는 만큼 때때로 상대의 사랑을 확인받고 싶어 하기도 합니다.')
					st.write('최고의 궁합: ESTP\n\n최악의 궁합: ENFP')
				with tab3:
					st.subheader('#용감한 :blue[수호자]')
					st.write('행정 보조원, 인사 관리자, 신용 상담가, 보호 감찰관, 물리치료사, 정신과 의사, 방사선 기사')
				with tab4:
					st.subheader('#ISFJ :blue[유명인]')
					st.write('여자 유명인: ITZY 예지, 문채원, 아이린, 트와이스 다현, 태연')
					st.write('남자 유명인: 갓세븐 진영, 뉴이스트 종현, 최강창민, 몬스터엑스 셔누, 이민혁')
			elif result=='ISFP':
				with tab1:
					st.subheader('#호기심 많은 :blue[예술가형]')
					st.write('호기심이 강하며 자율적이고 융통적인 성격이고 임기응변에 강하고 다양한 삶을 선호하는 유형입니다. 특징은 낯가림이 심하고 매사에 점잖은 스타일이며, 장점으로는 감각적이고 열정적, 독창적인 장점이 있으며,인간관계에서 조화롭고 배려 깊은 점입니다. 단점으로는 소심하고 거절을 못 하는 성격이라고 합니다.')
				with tab2:
					st.subheader('#세심하게 느끼는, :blue[감성적인 연애의 예술가!]')
					st.write('평소에는 연애에 관심이 없습니다. 그렇기에 사랑에 빠지기 전까지 엄청나게 신중한 태도를 고집합니다. 이러한 과정을 거쳐서 연애를 하기 때문에 진심을 받는 것을 좋아합니다. 공감력이 뛰어나 상대방의 취향까지 존중해주고 잘 맞춰주며 긍정 에너지를 전파하는 ISFP는 간혹 자신에게 소홀해질 때도 있으니 이 점을 주의하시길 바랍니다.')
					st.write('최고의 궁합: ENTJ\n\n최악의 궁합: ENTP')
				with tab3:
					st.subheader('#호기심 많은 :blue[예술가]')
					st.write('보석 세공사, 음향 디자이너, 만화가, 지질학자, 사육사, 수의사, 법률 비서, 약사')
				with tab4:
					st.subheader('#ISFP :blue[유명인]')
					st.write('여자 유명인: 나나, 슬기, 웬디, 설현, 쯔위')
					st.write('남자 유명인: 유승호, 유재석, 김종국, 권정열, BTS 정국')
			elif result=='ENFP':
				with tab1:
					st.subheader('#재기 발랄한 :blue[활동가형]')
					st.write('창의적이며 활발한 성격을 지녔으며,상상력이 풍부하며 사고가 자유롭습니다. ENFP 장점으로는, 창의적인 생각이 넘치기 때문에, 조별과제나 프로젝트에서 아이디어 뱅크 역할을 맡습니다. 외향적인 성격으로 어디를 가나 분위기 메이커 라고 합니다. 단점으로는, 호기심은 많지만,금방 지루함을 느끼기 때문에 집중력이나 끈기가 약하며, 감정 변화에 폭이 커 가끔 주변 사람들을 힘들게 한다고 합니다.')
				with tab2:
					st.subheader('#긍정적 에너지로 사랑을 불어넣는, :blue[연애의 최적화 전문가]')
					st.write('연인과 함께라면 언제, 어떤 상황이라도 OK인 낭만적인 사람입니다. 한번 사랑에 빠지면 화끈하게 올인하는데요. 상대방에게 헌신적인 연애를 할 가능성이 높기에 상대방 위주로 연애를 할 것이라고 오해하지만, 연인을 사랑하는 만큼 자기 자신도 사랑하기에 크게 휘둘리지 않는 편입니다.')
					st.write('최고의 궁합: INFJ\n\n최악의 궁합: ISFJ')
				with tab3:
					st.subheader('#재기 발랄한 :blue[활동가]')
					st.write('크리에이티브 디렉터, 디자이너, 시나리오 작가, 방송 프로듀서, 홍보 컨설턴트, 상담사, 상품 기획자')
				with tab4:
					st.subheader('#ENFP :blue[유명인]')
					st.write('여자 유명인: 권은비, 이효리, 카리나, 조보아, ITZY 유나')
					st.write('남자 유명인: 싸이, 노홍철, RM, 조세호, 송민호')
			elif result=='ENFJ':
				with tab1:
					st.subheader('#정의로운 :blue[사회운동가형]')
					st.write('정의로운 사회운동가라는 타이틀에 맞게, 신뢰, 설득력, 이타적, 경청, 지도자, 계획적 등의 이상적인 리더의 자질을 나타내고 있습니다. 반면, 추진력이 좋은 만큼 가끔 열정이 과다할 때가 있어,주변 사람들이 힘들어하는 경우도 있다고 합니다. 고집이 세고, 성급하다는 것이 특징입니다. ')
				with tab2:
					st.subheader('#깊은 이해와 배려로 사랑을 끌어내는, :blue[연애의 카리스마]')
					st.write('눈치가 빨라 연인의 감정을 잘 캐치하고 연인의 장점을 알아보고 응원해주는 데 능숙해서 든든한 지원군 역할을 해줍니다. 그러나 상처를 주는 것에도, 받는 것에도 부정적인 편이라 문제가 생긴다면 문제의 골이 깊어집니다. 모든 사람들에게 다정하다고 느낄 수 있지만, 내 사람에 대한 애정은 확실합니다.')
					st.write('최고의 궁합: ISTP\n\n최악의 궁합: ISTJ')
				with tab3:
					st.subheader('#정의로운 :blue[사회운동가]')
					st.write('아나운서, 리포터, 방송 MC, 언어교사, 아동 복지사, CEO, 취업 컨설턴트, 동시 통역가')
				with tab4:
					st.subheader('#ENFJ :blue[유명인]')
					st.write('여자 유명인: 신세경, 유이, ITZY 리아, 여자 아이들 우기, 원진아')
					st.write('남자 유명인: 정우성, 임시완, 윤시윤, 공명, 강다니엘')
			elif result=='ENTP':
				with tab1:
					st.subheader('#논쟁을 즐기는 :blue[변론가형]')
					st.write('끊임없이 새로운 것을 시도하며 도전적인 자세를 즐기고 활동적인 것을 좋아하며 변화를 두려워하지 않습니다. ENTP 특징으로는 고집이 세고, 솔직한 편이며, 아이디어는 넘치지만, 일을 제대로 마무리한 적이 없을 만큼, 충동적인 결정을 하는 타입입니다. 재치있는 성격에 창의적이고 독창적인 주관을 하고 있는 것이 장점이며 단점으로는 충동적인 의사결정 때문에 실수가 잦으며, 해보고 싶은 건 모두 해봐야 하는 성격 탓에 "반항아"라는 타이틀이 붙을 만큼 고집이 세다고 합니다.')
				with tab2:
					st.subheader('#새로운 사랑의 가능성을 찾아내는, :blue[연애의 혁신가]')
					st.write('개인의 시간을 중요시 여기며 자기애가 강한 편입니다. 자존감도 높고 상상력이 풍부해 계획 없이 어디론가 튈 수 있습니다. 호불호가 확실하고 싫증을 빨리 내는 편이라 새로운 핫플이나 액티비티를 함께 찾아다닐 수 있는 사람을 만나면 지루할 틈 없이 다이나믹한 연애를 즐길 수 있습니다. 그래서 매력적이지만 감정적일 거라고 생각하면 큰 오해입니다. 굉장히 논리적인 사고로 팩트를 지적하는 타입입니다.')
					st.write('최고의 궁합: ISFJ\n\n최악의 궁합: ISFP')
				with tab3:
					st.subheader('#논쟁을 즐기는 :blue[변론가]')
					st.write('발명가, 벤처 사업가, 에이전트, 배우, 가수, 영화감독, 칼럼니스트, 정치인')
				with tab4:
					st.subheader('#ENTP :blue[유명인]')
					st.write('여자 유명인: 한예슬, 김희선, 김세정, 강지영, 제시')
					st.write('남자 유명인: 이이경, 영탁, 육성재, 이찬혁, 몬스터엑스 민혁')
			elif result=='ENTJ':
				with tab1:
					st.subheader('#대담한 :blue[통솔자형]')
					st.write('철저하면서도 단호하며 결단력과 통솔력이 뛰어난 리더형 인물입니다. ENTJ 특징으로는 자기주장이 강하며, 솔직하고, "효율 빼면 시체"라고 할 정도로 효율적인 것을 좋아한다고 합니다. 주로 주도적이며 리더십이 있고, 본인에 대한 확신이 있어 매사에 자신감 있는 태도를 보이는 것이 장점입니다. 단점으로는 성격이 급하고, 다소 공격적으로 느껴질 수 있는 태도와 타인에 대한 공감능력이 부족한 점이라고 합니다.')
				with tab2:
					st.subheader('#목표 지향적인 사랑을 구현하는, :blue[연애의 CEO]')
					st.write('일에 있어서 리더십을 발휘하는 ENTJ는 연애에 있어서도 리더의 자질을 보입니다. 연인이 고민할 때 빠르게 해결책을 제시하고 싸울 때도 잘잘못을 확실히 따지기 때문입니다. 하지만 단호하고 이성적인 모습이 때론 연인을 가르치려는 것처럼 보여서 불화가 생기기도 합니다. 소유욕이 강한 편이라 상대방을 유혹하는 데 뛰어납니다.')
					st.write('최고의 궁합: ISFP\n\n 최악의 궁합: ISFJ')
				with tab3:
					st.subheader('#대담한 :blue[통솔자]')
					st.write('경영 컨설턴트, 공인 중개사, 관리사, 변호사, 재무 상담사, 경제 분석가, 벤처 투자가, 판사')
				with tab4:
					st.subheader('#ENTJ :blue[유명인]')
					st.write('여자 유명인: 소녀시대 티파니, 러블리즈 베이비소울, 소녀시대 서현, 이달의 소녀 희진, 프로미스나인 박지원')
					st.write('남자 유명인: 샤이니 키, 지코, 이특, 이승기, 정해인')
			elif result=='ESTP':
				with tab1:
					st.subheader('#모험을 즐기는 :blue[사업가형]')
					st.write('직관력이 뛰어나며 스스로 역량을 믿어 자신감 넘치게 모든 일을 수행하는 유형입니다. 변화에 잘 적응하는 인싸 기질이 있으며, 충동적이고 즉흥적인 것이 특징입니다. 대표적인 장점으로는 추진력이 있고, 힘이 넘친다는 점과 긍정적이고 주변 사람들을 잘 챙긴다는 점이 있습니다. 단점으로는 자기주장이 강하고, 고집이 세서 타인과의 의견 충돌이 생기는 경우가 잦은 점이 단점입니다.')
				with tab2:
					st.subheader('#브레이크 고장난 :blue[직진러버]')
					st.write('이상형을 만나면 어마어마한 사교성과 추진력으로 연애까지 빠르게 직진하는 스타일입니다. 남의 눈치를 보지 않고 애정표현이나 스킨십도 적극적으로 하는 편이며, 갈등이 생겨도 피하지 않아 뒤끝이 없습니다. 하지만 자기주장이 강해 충동적일 때가 있으며 원하는 무언가에 꽂히면 불도저 같은 모습을 보이기도 합니다.')
					st.write('최고의 궁합: INFJ\n\n최악의 궁합: INFP')
				with tab3:
					st.subheader('#모험을 즐기는 :blue[사업가]')
					st.write('경찰관, 소방관, 군 장교, 펀드 매니저, 은행원, 기자, 여행 가이드, 건축 엔지니어')
				with tab4:
					st.subheader('#ESTP :blue[유명인]')
					st.write('여자 유명인: 블랙핑크 지수, 이유비, 경리, 우주소녀 연정, 케플러 김채현')
					st.write('남자 유명인: 전현무, 조나단, NCT 성찬, NCT 재현,신동엽')
			elif result=='ESTJ':
				with tab1:
					st.subheader('#엄격한 :blue[관리자형]')
					st.write('모든 일을 계획적으로 잡고 실행하는 것을 좋아하는 유형. 목표를 위해 열정과 노력을 마다치 않으며 엄격한 관리자 유형이라고 합니다. 대표적인 ESTJ 특징으로는 매우 계획적이고,현실적인 문제를 명확하게 파악한다는 것입니다. 장점은 리더십,책임감, 의리가 있다는 점이며, 단점으로는, 지나친 원리원칙 주의와, 고집이 세서 타인의 말에 공감을 잘 못 한다는 점입니다.')
				with tab2:
					st.subheader('#신뢰성 있는 사랑을 추구하는, :blue[연애의 실행자]')
					st.write('경제적, 시간적인 개념 등 모든 면에서 계획적인 성향을 가진 철저한 스타일입니다. 밀당이라고는 전혀 모르며 선비 같다는 말을 종종 듣습니다. 책임감이 강해서 약속을 반드시 지키기 때문에 연인에게 금방 신뢰를 얻기도 합니다. 연애도 사랑도 열심히 학습하는 노력파이기에 시간이 지날수록 애정표현은 점점 좋아질 겁니다.')
					st.write('최고의 궁합: INFP\n\n최악의 궁합: INFJ')
				with tab3:
					st.subheader('#엄격한 :blue[관리자]')
					st.write('감독관, 예산 분석가, 은행장, 정책 책임자, 보안 요원, 기관사, 교육 전')
				with tab4:
					st.subheader('#ESTJ :blue[유명인]')
					st.write('여자 유명인: 한채영, 한가인, 남지현, 제시카, 이지혜')
					st.write('남자 유명인: 뱀뱀, 김준수, 류준열, 데프콘, 조규성')
			elif result=='ESFP':
				with tab1:
					st.subheader('#자유로운 영혼의 :blue[연예인형]')
					st.write('타인과의 관계를 중요시하기에 모든 이들에게 친절하고, 원만한 관계를 유지하는 유형이고, 즉흥적이고 에너지가 넘치고 사교적이고 유머러스하다는 것이 특징이다. ESFP 장점으로는 도전적이지만, 주체할 수 없는 인싸 기질로 매사에 자유롭고 독창적인 ESFP 특징이 진지하지 못하다는 모습을 보여주는 것이 단점이라고 합니다.')
				with tab2:
					st.subheader('#사랑의 즐거움을 추구하는, :blue[연애의 엔터테이너]')
					st.write('모든 사람들에게 사랑받는 자유로운 영혼으로 연인에게 무조건적으로 헌신합니다. 이벤트를 자주 기획하기도 하죠. 연인이 인생의 1순위이기 때문에 모든 MBTI를 통틀어 가장 열정적으로 사랑하는 스타일이라고 할 수 있습니다. 그래서인지 이별할 때 미련이 없는 편이기도 합니다.')
					st.write('최고의 궁합: INTJ\n\n최악의 궁합: INTP')
				with tab3:
					st.subheader('#자유로운 :blue[영혼의 연예인]')
					st.write('코미디언, 의상 디자이너, 일러스트레이터, 애니메이터, 여행 상품 기획자, 놀이 치료사')
				with tab4:
					st.subheader('#ESFP :blue[유명인]')
					st.write('여자 유명인: 박보영, 소녀시대 윤아, 소녀시대 수영, 웬디, 이수현')
					st.write('남자 유명인: 비, 사이먼 도미닉, 정준하, 이재욱, 헨리')
			elif result=='ESFJ':
				with tab1:
					st.subheader('#사교적인 :blue[외교관형]')
					st.write('타인과의 관계에 신경을 많이 쓰는 유형이고 인정받는 것에 굉장히 민감하다고 합니다. 주변 사람들과 두루두루 잘 지내고, 공감 능력과 리액션이 뛰어나 MBTI 중에서도 특히 인간관계가 좋은 편입니다. 장점 키워드로는 사교적, 호기심, 의사소통능력, 열정, 책임감 등이 있고, 단점 키워드로는 완벽주의, 이상주의, 융통성 부족 등이 있습니다.')
				with tab2:
					st.subheader('#상대방을 배려하는, :blue[연애의 따뜻한 수호천사]')
					st.write('누구에게나 친절하기에 상대방은 간혹 이것이 관심인지 원래 본인의 성격인지 헷갈리고는 합니다. 연인의 감정을 잘 읽고 맞춰주기에 연애할 때 갈등이 적습니다. 하지만 속마음은 연인의 행동에 간섭하고 싶어 합니다. 그저 갈등을 만들고 싶지 않아서 불만을 속에 쌓아두는 것뿐입니다.')
					st.write('최고의 궁합: INTP\n\n최악의 궁합: ENTJ')
				with tab3:
					st.subheader('#사교적인 :blue[외교관]')
					st.write('홍보 책임자, 호텔 지배인, 마케팅 책임자, 초등학교 교사, 특수 교사, 비서, 유치원 교사')
				with tab4:
					st.subheader('#ESFJ :blue[유명인]')
					st.write('여자 유명인: 혜리, 이지현, 장윤정, 선예, 박초롱')
					st.write('남자 유명인: 박보검, 황광희, BTS 제이홉, 손흥민, 규현')
		except:
			st.image(img_np,caption="업로드한 이미지",use_column_width=True)
			with st.spinner('AI가 당신의 외모를 분석중입니다...'):
				time.sleep(3)  # 예시로 3초 동안 로딩 중 표시 (실제 분석으로 대체 필요)
				st.error('얼굴을 감지하지 못했습니다! 다른사진을 이용해주세요!')
	col1, col2 = st.columns(2)
	with col1:
		st.components.v1.html(f"<center>{kakao_ad_code1}</center>", height=250, scrolling=False)
	with col2:
		st.components.v1.html(f"<center>{kakao_ad_code2}</center>", height=250, scrolling=False)
	st.components.v1.html(coupang_ad_code, scrolling=False)
	st.markdown('<a target="_blank" href="https://icons8.com/icon/7338/%EC%96%BC%EA%B5%B4-%EC%9D%B8%EC%8B%9D-%EC%8A%A4%EC%BA%94">얼굴 인식 스캔</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>', unsafe_allow_html=True)
if __name__ == "__main__":
    main()
