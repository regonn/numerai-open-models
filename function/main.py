import functions_framework
from google.cloud import batch_v1
from datetime import datetime

IMAGE_URI = "YOUR_DOCKER_IMAGE_URI"
PROJECT_ID = "YOUR_PROJECT_ID"
REGION = "YOUR_REGION"
JOB_NAME = "YOUR_JOB_NAME"


@functions_framework.http
def numerai_webhook(request):
    client = batch_v1.BatchServiceClient()
    runnable = batch_v1.Runnable()
    runnable.container = batch_v1.Runnable.Container()
    runnable.container.image_uri = IMAGE_URI

    task = batch_v1.TaskSpec()
    task.runnables = [runnable]

    resources = batch_v1.ComputeResource()
    resources.cpu_milli = 1000  # 1 vCPU
    resources.memory_mib = 1024  # 1 GiB
    task.compute_resource = resources

    task.max_retry_count = 2
    task.max_run_duration = "3600s"

    group = batch_v1.TaskGroup()
    group.task_count = 1
    group.task_spec = task

    policy = batch_v1.AllocationPolicy.InstancePolicy()
    policy.machine_type = "e2-standard-4"
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
    create_request.job_id = JOB_NAME + datetime.now().strftime("%Y%m%d%H%M%S")
    create_request.parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    client.create_job(create_request)
    return "OK"
