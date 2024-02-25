kubectl delete deploy song-recommender-deployment-xs90
kubectl delete deploy project2-client-xs90
kubectl delete service project2-client-service-xs90
kubectl delete service song-recommender-service-xs90
kubectl delete job project2-xs90-job
kubectl delete pvc project2-pvc-xs90
kubectl -n xs90 get pods