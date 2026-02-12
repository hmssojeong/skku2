import streamlit as st # 스트림릿 라이브러리를 읽어와서 st라고 부르겠다.

st.title("안녕하세요")
if st.button("노크하기"):
    st.write("여기 민규있어요~")

st.write("동의하시면 아래 내용에 체크해주세요.")
agree = st.checkbox("동의합니다.")
if agree:
    st.write("당신은 동의했습니다.")

option = st.selectbox("연락을 어떻게 받을래요?", ("이메일","전화","문자"))
st.write("선택한 방식: " + option)

age = st.slider("당신은 몇 살인가요?",0,100)
st.write("저는"+str(age)+"살 입니다!")

text_name = st.text_input("이름을 입력해주세요")
text_intro = st.text_area("자기소개 해주세요.")

st.image("https://picsum.photos/250/250")

st.video('https://youtu.be/zUGPcKGP21E?si=9gGdTL_GSgEhMqJg')

# streamlit run app.py