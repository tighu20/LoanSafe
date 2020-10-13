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

EXPOSE 8080


COPY main.py /
COPY embeddings.py /
COPY requirements.txt /

COPY saved_models /saved_models
COPY images /images
COPY data /data



RUN apt-get update
RUN pip install -r requirements.txt



CMD streamlit run --server.port 8080 --server.enableCORS false main.py

