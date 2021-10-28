import glob
import logging
import multiprocessing
import os
import ast
import shutil
import sqlite3
import datetime
from pathlib import PurePath

# Import smtplib for the actual sending function
import smtplib
import subprocess

# Import the email modules we'll need
import uuid
from email.message import EmailMessage
from zipfile import ZipFile

import imageio
import numpy as np
import tifffile
from flask import Flask, render_template, request
from google_drive_downloader import GoogleDriveDownloader as gdd
from validate_email import validate_email


class JobItem:
    def __init__(
        self,
        ID,
        submit_time,
        first_name,
        last_name,
        email,
        proj_name,
        status,
        download_url,
        train_cfg,
        description,
    ):
        self.ID = ID
        self.submit_time = submit_time
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.proj_name = proj_name
        self.status = status
        self.download_url = download_url
        self.train_cfg = train_cfg
        self.description = description

    @staticmethod
    def from_request(req):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return JobItem(
            str(uuid.uuid4()),
            now,
            req["first-name"],
            req["last-name"],
            req["email-addr"],
            req["proj-name"],
            "SUBMIT",
            req["gdrive-url"],
            {
                "epochs": int(req["epochs"]),
                "batch_size": int(req["batch-size"]),
                "learning_rate": float(req["learning-rate"]) * 1.0e-3,
                "weight_decay": float(req["weight-decay"]) * 1.0e-5,
                "drop_rate": float(req["drop-rate"]),
                "n_models": int(req["n-models"]),
            },
            req['short-description'],
        )

    def __str__(self):
        return (
            f"[{self.status}] ID: {self.ID}, Name: {self.first_name} {self.last_name}"
            + f" ({self.email}), URL: {self.download_url}, config: {self.train_cfg}"
        )

    def __repr__(self):
        return self.__str__()


class SQL:
    def __init__(self, sqlite_f):
        self.conn = sqlite3.connect(sqlite_f)

    def create_table(self):
        command = """CREATE TABLE job_list
                       (ID text, timestamp text, first_name text, last_name text, email text, proj_name text, status text, download_url text, description text, train_cfg text)"""
        self.conn.execute(command)
        self.conn.commit()

    def get_jobs(self):
        job_list = []
        for row in self.conn.execute(
            """SELECT * FROM job_list ORDER BY timestamp DESC"""
        ):
            job_list.append(JobItem(*row[:-1], ast.literal_eval(row[-1])))
        return job_list

    def insert_job(self, job_item: JobItem):
        command = f"INSERT INTO job_list VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        self.conn.execute(
            command,
            (
                job_item.ID,
                job_item.submit_time,
                job_item.first_name,
                job_item.last_name,
                job_item.email,
                job_item.proj_name,
                job_item.status,
                job_item.download_url,
                job_item.description,
                str(job_item.train_cfg),
            ),
        )
        self.conn.commit()

    def update_job_status(self, job_id: str, new_status: str):
        self.conn.execute(
            f"""UPDATE job_list SET status = '{new_status}' WHERE ID = '{job_id}' """
        )
        self.conn.commit()

    def update_job_status_with_url(self, job_id: str, new_status: str, url: str):
        assert (
            new_status == "SUCCESS"
        ), "Only be able to post download url when status == SUCCESS"
        self.conn.execute(
            f"""UPDATE job_list SET status = '{new_status}', download_url = '{url}' WHERE ID = '{job_id}' """
        )
        self.conn.commit()


SQL_FILE = "./job_list.sqlite"
if not os.path.isfile(SQL_FILE):
    SQL_HANDLE = SQL(SQL_FILE)
    SQL_HANDLE.create_table()
    job_list = [
            JobItem(str(uuid.uuid4()), "2019-01-01 09:00:00", "Xuanqing", "Liu", "xqliu@cs.ucla.edu", "Test job list", "SUBMIT", "TBD", {"batch_size": 4, "learning_rate": 1.0e-3}, "DESC"),
            JobItem(str(uuid.uuid4()), "2019-01-01 09:00:01", "Xuanqing", "Liu", "xqliu@cs.ucla.edu", "Test job list", "QUEUE", "TBD", {"batch_size": 4, "learning_rate": 1.0e-3}, "DESC"),
            JobItem(str(uuid.uuid4()), "2019-01-01 09:00:02", "Xuanqing", "Liu", "xqliu@cs.ucla.edu", "Test job list", "RUN", "TBD", {"batch_size": 4, "learning_rate": 1.0e-3}, "DESC"),
            JobItem(str(uuid.uuid4()), "2019-01-01 09:00:03", "Xuanqing", "Liu", "xqliu@cs.ucla.edu", "Test job list", "SUCCESS", "TBD",  {"batch_size": 4, "learning_rate": 1.0e-3}, "DESC"),
            JobItem(str(uuid.uuid4()), "2019-01-01 09:00:04", "Xuanqing", "Liu", "xqliu@cs.ucla.edu", "Test job list", "FAILURE", "TBD", {"batch_size": 4, "learning_rate": 1.0e-3}, "DESC"),
    ]
    for job in job_list:
        SQL_HANDLE.insert_job(job)

app = Flask(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def send_email(title, body, recipient):
    """Send email to somebody"""
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = title
    msg["From"] = "ai.reporter@yahoo.com"
    msg["To"] = recipient
    src_email = "ai.reporter@yahoo.com"
    # this password is only for sending & receiving emails. Not for account login
    password = "dxztrlarxrfpistq"
    logging.info(f"Sending email from {src_email} to {recipient}")
    with smtplib.SMTP_SSL(host="smtp.mail.yahoo.com", port=465, timeout=15) as server:
        print("logging in")
        server.set_debuglevel(1)
        server.login(src_email, password)
        server.helo()
        server.send_message(msg)


def confirmation_mail_body(job: JobItem, num_training_imgs, num_testing_imgs):
    """Compose the body of confirmation email.
    :param result: the result of submitted form
    :param num_training_imgs:  Number of training images
    :param num_testing_imgs: Number of testing images
    :return: str, the body of email
    """
    lines = []
    lines.append(f"{job.first_name}:")
    lines.append(
        "This is to confirm that we have successfully downloaded and analyzed your files. Below are some statistics:"
    )
    lines.append(f"Project name: {job.proj_name}")
    lines.append(f"Google drive URL: {job.download_url}")
    lines.append(f"#Training images: {num_training_imgs:,}")
    lines.append(f"#Testing images: {num_testing_imgs:,}")
    lines.append(f"Experiment description: {job.description}")
    # How to calculate the expected time?
    hours = num_testing_imgs / 800 * 2
    lines.append(f"Expected time to finish: {hours:.2f} hours")
    lines.append(
        "You will receive another email containing downloadable links to model checkpoints, predictions, etc."
    )
    lines.append("\n\nBest regards,")
    lines.append("AI Reporter team")
    return "\n".join(lines)


def results_mail_body(result):
    lines = []
    lines.append(f"{result['first-name']}:")
    lines.append(
        "This is to notify that the job is finished and the model/predictions are "
        "ready. You can retrieve them at https://xxxx.com"
    )
    _, passcode = get_and_verify_gdrive_url(result["goog_uri"])
    lines.append(f"using following passcode: {passcode}")


def train_marker(src, tgt, train_cfg, ckpt_dir):
    ret_code = 0
    for k_model in range(train_cfg['n_models']):
        ckpt_f = os.path.join(ckpt_dir, f"model_{k_model}")
        cmd = (
            f"CUDA_VISIBLE_DEVICES=0 python -m BNNBench.trainer.ensemble_trainer --src-dir {src} --tgt-dir {tgt} "
            f"--dropout --batch-size {train_cfg['batch_size']} --img-size 1024 "
            f"--init-lr {train_cfg['learning_rate']} --total-epochs {train_cfg['epochs']} "
            f"--weight-decay {train_cfg['weight_decay']} --ckpt-file {ckpt_f}"
        )
        ret_code = subprocess.run(cmd, shell=True)
        logging.info(f"Training code finished with return code ({ret_code})")
    return ret_code


def test_marker(src, tgt, ckpt_dir, result_dir):
    cmd = (
        f"CUDA_VISIBLE_DEVICES=0 python -m BNNBench.inference.ensemble_inference --ckpt-files {ckpt_dir}/* "
        f"--img-size 1024 --batch-size 16 --test-src-dir {src} --test-tgt-dir {tgt} "
        f"--result-dir {result_dir}"
    )
    ret_code = subprocess.run(cmd, shell=True)
    logging.info(f"Testing code finished with return code ({ret_code})")
    return ret_code


def pack_result_and_upload(result_dir):
    result_tar_ball = os.path.join("./results", PurePath(result_dir).stem + ".tar.gz")
    pack_cmd = f"tar -czvf {result_tar_ball} {result_dir}/"
    ret_code = subprocess.run(pack_cmd, shell=True)
    logging.info(f"Created tar ball with return code {ret_code}")
    if ret_code.returncode == 0:
        # upload to backblaze b2, currently we are using Xuanqing's personal account.
        # In the future we need to have a formal account
        tar_f = os.path.basename(result_tar_ball)
        upload_cmd = f"b2 upload-file MSC-project {result_tar_ball} fnet-web/{tar_f}"
        with subprocess.Popen(upload_cmd, shell=True, stdout=subprocess.PIPE) as proc:
            # read first line for download url
            url = proc.stdout.readline().decode("utf8").lstrip("URL by file name: ")
        logging.info(f"Uploaded tar ball with downloading url {url}")
    return url

def process_task(job: JobItem, file_id):
    # update state
    SQL_HANDLE = SQL(SQL_FILE)
    SQL_HANDLE.update_job_status(job.ID, "RUN")
    try:
        logging.info(f"Downloading {file_id}")
        unzip_dest = f"third_party_data/{file_id}"
        zip_dest = unzip_dest + ".zip"
        # unzip to file_id folder
        if not os.path.isfile(zip_dest):
            gdd.download_file_from_google_drive(
                file_id=file_id,
                dest_path=zip_dest,
                unzip=False,
                showsize=True,
                overwrite=True,
            )
            with ZipFile(zip_dest, "r") as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(unzip_dest)
        # extract the root folder name
        unzip_dest = glob.glob(f"{unzip_dest}/*/")[0]
        # move images to a separate folder namely A and B
        train_src_dir = os.path.join(unzip_dest, "input/train")
        train_tgt_dir = os.path.join(unzip_dest, "output/train")
        test_src_dir = os.path.join(unzip_dest, "input/test")
        test_tgt_dir = os.path.join(unzip_dest, "output/test")
        logging.info("Building training and testing folders")
        num_training_imgs = len(os.listdir(train_src_dir))
        num_testing_imgs = len(os.listdir(test_src_dir))
        send_email(
            "Confirmation email from AI-reporter",
            confirmation_mail_body(job, num_training_imgs, num_testing_imgs),
            job.email,
        )
        # train for this marker
        logging.info("Launching the training loop")
        ckpt_dir = os.path.join("./checkpoints", job.ID)
        os.makedirs(ckpt_dir, exist_ok=True)
        train_marker(train_src_dir, train_tgt_dir, job.train_cfg, ckpt_dir)
        # generate the prediction result
        logging.info("Generating the prediction result")
        result_dir = os.path.join("./results", job.ID)
        os.makedirs(result_dir, exist_ok=True)
        test_marker(test_src_dir, test_tgt_dir, ckpt_dir, result_dir)
        # package the result and upload
        result_url = pack_result_and_upload(result_dir)
        # Send email to client
        send_email("Your model and predictions are ready", f"Your model and predictions can be downloaded from {result_url}", job.email)
    except Exception as e:
        send_email("JOB FAILURE", str(e), job.email)
        SQL_HANDLE.update_job_status(job.ID, "FAILURE")
        raise e
    SQL_HANDLE.update_job_status_with_url(job.ID, "SUCCESS", result_url)


def get_and_verify_gdrive_url(url):
    parts = url.split("/")
    try:
        idx_d = parts.index("d")
        file_id = parts[idx_d + 1]
        is_valid_uri = True
    except:
        is_valid_uri = False
        file_id = None
    return is_valid_uri, file_id


@app.route("/start_training", methods=["POST"])
def start_training():
    result = request.form
    logging.info(f"{result}")
    job = JobItem.from_request(result)
    is_valid_uri, file_id = get_and_verify_gdrive_url(job.download_url)
    is_valid_email = validate_email(job.email)
    if is_valid_email and is_valid_uri:
        # start a new process to download this file
        download_thread = multiprocessing.Process(
            target=process_task,
            args=(job, file_id),
            name=f"Download for {job.email}",
        )
        download_thread.start()
    else:
        logging.info(f"Got an invalid input: {job.email}, {job.download_url}")

    # insert a new entry to database
    SQL_HANDLE = SQL(SQL_FILE)
    SQL_HANDLE.insert_job(job)
    return render_template(
        "success_page.html",
        result=result,
        valid_email=is_valid_email,
        valid_uri=is_valid_uri,
    )


@app.route("/show_jobs")
def show_jobs():
    SQL_HANDLE = SQL(SQL_FILE)
    job_list = SQL_HANDLE.get_jobs()
    return render_template("show_jobs.html", job_list=job_list)


@app.route("/help")
def help():
    return render_template("help.html")


@app.route("/")
def hello_world():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="localhost", port=8991, debug=True)
# download_uri("1FBr8UjM6bMwidgmg_qyyvklCqmdK5Zkt", "xqliu@ucdavis.edu")
