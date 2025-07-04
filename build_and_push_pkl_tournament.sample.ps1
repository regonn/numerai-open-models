$GIT_COMMIT = git rev-parse HEAD
$REGION = "asia-northeast1"
$PROJECT_ID = "YOUR_GCP_PROJECT_ID"
$ARTIFACT_REGISTRY_REPOSITORY = "YOUR_ARTIFACT_REGISTRY_REPOSITORY_NAME"
$MODEL_ID = "YOUR_MODEL_ID"

docker build --build-arg NUMERAI_MODEL_ID=${MODEL_ID} --build-arg GIT_COMMIT=$GIT_COMMIT --progress=plain -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPOSITORY}/${MODEL_ID}:latest -f Dockerfile.pkl_tournament .
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPOSITORY}/${MODEL_ID}:latest