# Numerai Open Models

- /models
  - model files for google cloud batch
- /function
  - code for google cloud function(webhook trigger)

1. Please create a repository in GCP Artifact Registry.
1. Build your code with Docker and push it to the repository you've created. The code in build_and_push.sh may serve as a reference.
1. Create a Google Cloud Function that utilizes the Docker image from the repository you've created to launch jobs in Google Cloud Batch. Depending on the GCP instance you specify, it's possible to increase memory and GPU resources.

## Support me

https://debank.com/profile/0xad3ab30bf7e7423fa8266c7300f2dcd07377dc7e
