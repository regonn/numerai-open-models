import os
from datetime import datetime

from dotenv import load_dotenv
from google.cloud import batch_v1
from google.oauth2 import service_account

load_dotenv()
MODEL_ID = os.environ.get("BATCH_MODEL_ID")
IMAGE_URI = os.environ.get("BATCH_IMAGE_URI_BASE") + MODEL_ID
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = os.environ.get("BATCH_REGION")
JOB_NAME = f"numerai-predict-batch-{MODEL_ID}-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def run_batch():
    # Set up Google Cloud credentials
    credentials = service_account.Credentials.from_service_account_info(
        {
            "type": "service_account",
            "project_id": os.environ.get("GCP_PROJECT_ID"),
            "private_key_id": os.environ.get("GCS_PRIVATE_KEY_ID"),
            "private_key": os.environ.get("GCS_PRIVATE_KEY").replace("\\n", "\n"),
            "client_email": os.environ.get("GCS_CLIENT_MAIL"),
            "client_id": os.environ.get("GCS_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.environ.get("GCS_CLIENT_X509_CERT_URL"),
        }
    )

    client = batch_v1.BatchServiceClient(credentials=credentials)
    runnable = batch_v1.Runnable()
    runnable.container = batch_v1.Runnable.Container()
    runnable.container.image_uri = IMAGE_URI

    task = batch_v1.TaskSpec()
    task.runnables = [runnable]

    environment = batch_v1.Environment()
    environment.variables = {
        "DISCORD_WEBHOOK_URL": os.environ.get("DISCORD_WEBHOOK_URL", ""),
        "NUMERAI_PUBLIC_ID": os.environ.get("NUMERAI_PUBLIC_ID", ""),
        "NUMERAI_SECRET_KEY": os.environ.get("NUMERAI_SECRET_KEY", ""),
        "NUMERAI_MODEL_ID": os.environ.get("NUMERAI_MODEL_ID", ""),
        "GCP_PROJECT_ID": os.environ.get("GCP_PROJECT_ID", ""),
        "GCS_PRIVATE_KEY_ID": os.environ.get("GCS_PRIVATE_KEY_ID", ""),
        "GCS_PRIVATE_KEY": os.environ.get("GCS_PRIVATE_KEY", ""),
        "GCS_CLIENT_MAIL": os.environ.get("GCS_CLIENT_MAIL", ""),
        "GCS_CLIENT_ID": os.environ.get("GCS_CLIENT_ID", ""),
        "GCS_CLIENT_X509_CERT_URL": os.environ.get("GCS_CLIENT_X509_CERT_URL", ""),
        "GCS_BUCKET_NAME": os.environ.get("GCS_BUCKET_NAME", ""),
    }
    task.environment = environment

    resources = batch_v1.ComputeResource()
    resources.cpu_milli = 16000  # 16 vCPU
    resources.memory_mib = 131072  # 128 GB
    task.compute_resource = resources

    task.max_retry_count = 0
    task.max_run_duration = "3600s"  # Daily Submission Limit

    group = batch_v1.TaskGroup()
    group.task_count = 1
    group.task_spec = task

    policy = batch_v1.AllocationPolicy.InstancePolicy()
    policy.machine_type = "c4-highmem-16"
    instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
    instances.policy = policy
    allocation_policy = batch_v1.AllocationPolicy()
    allocation_policy.instances = [instances]

    job = batch_v1.Job()
    job.task_groups = [group]
    job.allocation_policy = allocation_policy

    job.logs_policy = batch_v1.LogsPolicy()
    job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING

    create_request = batch_v1.CreateJobRequest()
    create_request.job = job
    create_request.job_id = JOB_NAME
    create_request.parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    client.create_job(create_request)
    return "OK"


if __name__ == "__main__":
    run_batch()
