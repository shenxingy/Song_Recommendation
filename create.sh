kubectl -n xs90 apply -f deployment.yaml
kubectl -n xs90 apply -f service.yaml
# wait for the pods to be ready
sleep 20
kubectl -n xs90 get pods