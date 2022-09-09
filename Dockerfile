FROM python:3.8

COPY server.py ./
COPY inference.py ./
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_HOST 0.0.0.0
EXPOSE 5000

CMD ["python", "server.py"]