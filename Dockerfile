FROM python:3.7
WORKDIR /idp_flask
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . /idp_flask 
ENTRYPOINT ["streamlit", "run"]
CMD ["app_streamlit.py"]