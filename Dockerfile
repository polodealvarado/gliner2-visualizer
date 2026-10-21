# Hugging Face Docker Space: CPU-only torch; the model is downloaded and checked at build time.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Dependencies live in the PEP 723 header of server.py. Torch comes from the CPU index to keep the image small.
COPY --chown=user server.py ./
RUN python -c "import re; print('\n'.join(re.findall(r'^#\s+\"(.+)\",$', open('server.py').read(), re.M)))" > requirements.txt \
 && pip install --user --no-cache-dir "$(grep '^torch' requirements.txt)" --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user --no-cache-dir -r requirements.txt
RUN python server.py --check

COPY --chown=user index.html ./
EXPOSE 7860
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "7860"]
