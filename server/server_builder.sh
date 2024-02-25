docker build -t xs90_serv:0.3 .
# docker build --no-cache -t xs90_serv:0.2 .
docker tag xs90_serv:0.3 xingyushen/xs90_serv:0.3
docker push xingyushen/xs90_serv:0.3
