docker build -t xs90_cli:0.7 .
# docker build --no-cache -t xs90_cli:0.1 .
docker tag xs90_cli:0.7 xingyushen/xs90_cli:0.7
docker push xingyushen/xs90_cli:0.7
