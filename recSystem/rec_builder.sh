docker build -t xs90_rec:0.3 .
# docker build --no-cache -t xs90_rec:0.1 .
docker tag xs90_rec:0.3 xingyushen/xs90_rec:0.3
docker push xingyushen/xs90_rec:0.3