import glob
import logging
import multiprocessing
import os
import shutil
import sqlite3
import datetime

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
                "epochs": req["epochs"],
                "batch_size": req["batch-size"],
                "learning_rate": req["learning-rate"],
                "weight_decay": req["weight-decay"],
                "drop_rate": req["drop-rate"],
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
                       (ID text, timestamp text, first_name text, last_name text, email text, proj_name text, status text, download_url text, train_cfg text, description text)"""
        self.conn.execute(command)
        self.conn.commit()

    def get_jobs(self):
        job_list = []
        for row in self.conn.execute(
            """SELECT * FROM job_list ORDER BY timestamp DESC"""
        ):
            job_list.append(JobItem(*row[:-1], eval(row[-1])))
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
                str(job_item.train_cfg),
                job_item.description,
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

"""
SQL_HANDLE = SQL(SQL_FILE)
SQL_HANDLE.create_table()
job_list = [
        JobItem(str(uuid.uuid4()), "2019-01-01 09:00:00", "Xuanqing", "Liu", "xqliu@cs.ucla.edu", "Test job list", "SUBMIT", "TBD", {"batch_size": 4, "learning_rate": 1.0e-3}, ""),
        JobItem(str(uuid.uuid4()), "2019-01-01 09:00:01", "Xuanqing", "Liu", "xqliu@cs.ucla.edu", "Test job list", "QUEUE", "TBD", {"batch_size": 4, "learning_rate": 1.0e-3}, ""),
        JobItem(str(uuid.uuid4()), "2019-01-01 09:00:02", "Xuanqing", "Liu", "xqliu@cs.ucla.edu", "Test job list", "RUN", "TBD", {"batch_size": 4, "learning_rate": 1.0e-3}, ""),
        JobItem(str(uuid.uuid4()), "2019-01-01 09:00:03", "Xuanqing", "Liu", "xqliu@cs.ucla.edu", "Test job list", "SUCCESS", "TBD", {"batch_size": 4, "learning_rate": 1.0e-3}, ""),
        JobItem(str(uuid.uuid4()), "2019-01-01 09:00:04", "Xuanqing", "Liu", "xqliu@cs.ucla.edu", "Test job list", "FAILURE", "TBD", {"batch_size": 4, "learning_rate": 1.0e-3}, ""),
]
for job in job_list:
    SQL_HANDLE.insert_job(job)
exit(0)
"""

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


def convert_to_png(in_img):
    """Convert the input image to png format. As png is known to work with pix2pix.
    We could also modify their code to directly support other formats as well.

    Input channels will be compressed to one channel. If input is color image, it will
    be converted to black and white image.

    Output image coexists with input image, only the file extension will be changed to
    png. If the input is already in png format, then it will be overwritten.
    :param in_img: Path to input image.
    :return: Path to
    """
    if in_img.endswith(".tiff") or in_img.endswith(".tif"):
        dat = tifffile.imread(in_img).astype(np.float)
    else:
        dat = imageio.imread(in_img).astype(np.float)
    if len(dat.shape) != 2:
        # guess the data spec
        if dat.shape[-1] == 3:
            # Convert rgb to grey
            dat = 0.2989 * dat[:, :, 0] + 0.5870 * dat[:, :, 1] + 0.1140 * dat[:, :, 2]
        elif dat.shape[-1] == 1:
            dat = dat[:, :, 0]
        else:
            raise RuntimeError("Unknown data structure: ", dat.shape, dat)
    assert dat.shape[0] == dat.shape[1] == 1024
    dat = dat.astype(np.uint8)
    # replace the file extension to png
    out_img = ".".join(in_img.split(".")[:-1] + ["png"])
    imageio.imwrite(out_img, dat)
    return out_img


def collect_images_and_copy(train_dir, test_dir, out_dir):
    """Collect images from training and testing directories, and move them to
    output directory for launching Pix2pix.
    :param train_dir: str, the root directory of training files.
    :param test_dir: str, the root directory of testing files.
    :param out_dir: str, the output directory in which both training and test
    files will be written.
    :return: the number of training images and the number of testing images.
    """

    def collect_images(folder):
        """Get all images under this folder with valid extension.
        :param folder: Path to directory containing images.
        :return: A list of path to images with sorted order.
        """
        images = []
        for valid_ext in ["png", "jpg", "jpeg", "tif", "tiff"]:
            images.extend(glob.glob(f"{folder}/*.{valid_ext}"))
        return sorted(images)

    def move_or_copy_overwrite(src_f, dst_folder, move=True):
        """A helper function to move/copy image from one place to another. The
        image file name will not change after moving/copying. It may also overwrite
        the file in dst_folder having the same file name.
        :param src_f: Path to source image.
        :param dst_folder: Path to target folder.
        :param move: Whether to move or copy the image.
        :return: None
        """
        dst_file = os.path.join(dst_folder, os.path.basename(src_f))
        if move:
            shutil.move(src_f, dst_file)
        else:
            shutil.copy(src_f, dst_file)

    train_signal_images = collect_images(f"{train_dir}/**/Signal/")
    train_target_images = [f.replace("Signal", "Target") for f in train_signal_images]
    train_A = os.path.join(out_dir, "A/train")
    train_B = os.path.join(out_dir, "B/train")
    os.makedirs(train_A, exist_ok=True)
    os.makedirs(train_B, exist_ok=True)
    for fa, fb in zip(train_signal_images, train_target_images):
        # Temperary!!
        # fb = fb.replace('WHITE', 'F1')
        assert os.path.isfile(fb)
        # convert image to png format
        fa = convert_to_png(fa)
        fb = convert_to_png(fb)
        move_or_copy_overwrite(fa, train_A)
        move_or_copy_overwrite(fb, train_B)

    # prepare the test folder, this is optional.
    test_signal_images = collect_images(f"{test_dir}/**/Signal/")
    test_A = os.path.join(out_dir, "A/test")
    test_B = os.path.join(out_dir, "B/test")
    os.makedirs(test_A, exist_ok=True)
    os.makedirs(test_B, exist_ok=True)
    for fa in test_signal_images:
        fa = convert_to_png(fa)
        move_or_copy_overwrite(fa, test_A, move=False)
        # a fake target image
        move_or_copy_overwrite(fa, test_B)

    # Gather some statistics
    return len(train_signal_images), len(test_signal_images)


def combine_A_and_B(out_dir):
    bash_cmd = (
        f"python ./scripts/combine_A_and_B.py --fold_A={out_dir}/A "
        f"--fold_B={out_dir}/B --fold_AB={out_dir}/AB"
    )
    ret_code = subprocess.run(bash_cmd.split())
    logging.info(f"Combine A and B script finished with return code ({ret_code})")
    return ret_code


"""
def train_marker(out_dir, expr_name):
    bash_cmd = (
        f"python ./train.py --dataroot={out_dir}/AB "
        f"--name {expr_name} "
        f"--gpu_ids 0 --model=pix2pix --input_nc=1 "
        f"--output_nc=1 --dataset_mode aligned --batch_size 2 "
        f"--load_size=1024 --crop_size 1024"
    )
    ret_code = subprocess.run(bash_cmd.split())
    logging.info(f"Training code finished with return code ({ret_code})")
    return ret_code


def test_marker(out_dir, results_dir, expr_name):
    bash_cmd = (
        f"python test.py --dataroot {out_dir}/AB "
        f"--name {expr_name} --gpu_ids 0 --model pix2pix --input_nc 1 "
        f"--output_nc 1 --dataset_mode aligned --load_size 1024 "
        f"--crop_size 1024 --results_dir {results_dir}"
    )
    ret_code = subprocess.run(bash_cmd.split())
    logging.info(f"Testing code finished with return code ({ret_code})")
    return ret_code
"""

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
    cmd = (
        f"python -m BNNBench.trainer.ensemble_trainer --src-dir {src} --tgt-dir {tgt}"
        f"--dropout {train_cfg['drop_rate']} --batch-size {train_cfg['batch_size']}"
        f"--img-size 1024 --init-lr {train_cfg['learning_rate']} --total-epochs {train_cfg['epochs']}"
        f"--weight-decay {train_cfg['weight_decay']} --ckpt-file {ckpt_dir}"
    )
    ret_code = subprocess.run(cmd.split())
    logging.info(f"Training code finished with return code ({ret_code})")
    return ret_code


def test_marker(src, tgt, ckpt_dir, result_dir):
    cmd = (
        f"python -m BNNBench.inference.ensemble_inference --ckpt-files {ckpt_dir}/* "
        f"--img-size 1024 --batch-size 16 --test-src-dir {src} --test-tgt-dir {tgt} "
        f"--result-dir {result_dir}"
    )
    ret_code = subprocess.run(cmd.split())
    logging.info(f"Testing code finished with return code ({ret_code})")
    return ret_code


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
        train_marker(train_src_dir, train_tgt_dir, job.train_cfg, job.ID)

        # generate the prediction result
        logging.info("Generating the prediction result")
        test_marker(out_dir, results_dir, expr_name)
        # Send email to client
        # send_email("Your model and predictions are ready")
    except Exception as e:
        send_email("JOB FAILURE", str(e), job.email)
        SQL_HANDLE.update_job_status(job.ID, "FAILURE")
        raise e
    SQL_HANDLE.update_job_status_with_url(job.ID, "SUCCESS", "")


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
    is_valid_email = validate_email(
        email_address=job.email, check_regex=True, check_mx=False
    )
    if is_valid_email and is_valid_uri:
        # start a new process to download this file
        download_thread = multiprocessing.Process(
            target=process_task,
            args=(job.ID, result, file_id, job.email),
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
