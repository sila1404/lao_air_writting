FROM ghcr.io/prefix-dev/pixi:0.43.3

WORKDIR /app

COPY . .

RUN pixi install -e api --locked

EXPOSE 8000

CMD [ "pixi", "run", "api", "--host", "0.0.0.0", "--port", "8000" ]