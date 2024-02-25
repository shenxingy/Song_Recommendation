docker build -t xs90_cli:0.4 .
# docker build --no-cache -t xs90_cli:0.1 .
docker tag xs90_cli:0.4 xingyushen/xs90_cli:0.4
docker push xingyushen/xs90_cli:0.4
