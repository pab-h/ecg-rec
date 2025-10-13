import logging
import logging.config

import os
import re
import requests
import shutil

from sys     import exit
from zipfile import ZipFile
from dotenv  import load_dotenv


# Log config

logging.config.fileConfig('logging.conf')

logger = logging.getLogger()


# Loading env

load_dotenv()

# Download configs

DATA_FOLDER         = os.environ.get("DATA_FOLDER") 
CODE15_LINKS_FILE   = os.environ.get("CODE15_LINKS_FILE") 
FILES_TO_DOWNLOAD   = int(os.environ.get("FILES_TO_DOWNLOAD")) 
DOWNLOAD_CHUNK_SIZE = int(os.environ.get("DOWNLOAD_CHUNK_SIZE")) 

# Checking data folder 

logger.info("Checking if data folder already exists")

if not os.path.exists(DATA_FOLDER):

    logger.warning("The data folder does not exist. Creating")
    os.makedirs(DATA_FOLDER)


# Checking if data cached

logger.info("Checking if data already downloaded")

if len(os.listdir(DATA_FOLDER)):

    logger.info("Everything is calm")
    exit()

logger.warning("Don't have the data. Downloading")

# Downloading the data

with open(CODE15_LINKS_FILE) as file:

    code15Links = file.readlines()
    code15Links = map(lambda link: link[: -1], code15Links)
    code15Links = list(code15Links)

logger.info(f"Files to download: {FILES_TO_DOWNLOAD}")
code15Links = code15Links[: FILES_TO_DOWNLOAD]

filenameMask = r'/files/([^/?]+)'

code15Filenames = map(
        lambda file: re
                    .search(filenameMask, file)
                    .group(1),
        code15Links
)
code15Filenames = map(
    lambda file: file.split(".")[0],
    code15Filenames
)
code15Filenames = list(code15Filenames)

for filename, link in zip(code15Filenames, code15Links):

    try: 
        logger.info(f"Start download file {link}")

        response = requests.get(link, stream = True)
        response.raise_for_status()

        zipPath  = os.path.join(DATA_FOLDER, filename)
        zipPath += ".zip"

        with open(zipPath, 'wb') as file:
            for chuck in response.iter_content(chunk_size = DOWNLOAD_CHUNK_SIZE):
                file.write(chuck)

        logger.info(f"File '{filename}'.zip downloaded successfully")

    except requests.exceptions.RequestException as e:
        logger.exception(f"Error during download of {filename}: {e}")
        os.remove(f"{DATA_FOLDER}/{filename}")
        exit()

# Unzip the data

logger.info("Unziping the data")

for filename in code15Filenames:

    zipPath = os.path.join(DATA_FOLDER, f"{filename}.zip")
    outPath = os.path.join(DATA_FOLDER, f"{filename}.hdf5")

    logger.info(f"Unziping: data/{filename}.zip -> data/{filename}.hdf5")

    with ZipFile(zipPath) as zipObject:
        with zipObject.open(f"{filename}.hdf5") as src, open(outPath, "wb") as dst:
            shutil.copyfileobj(src, dst)


# Cleaning up the waste

logger.info("Cleaning up the waste")

for filename in code15Filenames:

    logger.info(f"Cleaning {filename}.zip")

    zipPath = os.path.join(DATA_FOLDER, f"{filename}.zip")

    os.remove(zipPath)
