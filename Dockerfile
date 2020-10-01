FROM python:3.7

RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

EXPOSE 8501


COPY main.py /
COPY embeddings.py /
COPY requirements.txt /
COPY deep_nn /deep_nn
COPY plain_nn /plain_nn
COPY BigML_Dataset1.csv /
COPY file1.csv /

RUN apt-get update
RUN pip install -r requirements.txt

CMD ["streamlit", "run","main.py"]

