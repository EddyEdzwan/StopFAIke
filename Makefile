# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* StopFAIke/*.py

black:
	@black scripts/* StopFAIke/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr StopFAIke-*.dist-info
	@rm -fr StopFAIke.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


# ----------------------------------
#      GCP
# ----------------------------------
# path of the file to upload to gcp (the path of the file should be absolute or should match the directory where the make command is run)
# LOCAL_PATH=/Users/julienseguy/code/EddyEdzwan/StopFAIke/raw_data/FakesNewsNET.csv
# LOCAL_PATH=/Users/julienseguy/code/EddyEdzwan/StopFAIke/raw_data/kaggle_Bisaillon/Fake.csv
# LOCAL_PATH=/Users/julienseguy/code/EddyEdzwan/StopFAIke/raw_data/kaggle_Bisaillon/True.csv
# LOCAL_PATH=/Users/julienseguy/code/EddyEdzwan/StopFAIke/raw_data/ISOT_Fake_News_Dataset/ISOT_Fake.csv
# LOCAL_PATH=/Users/julienseguy/code/EddyEdzwan/StopFAIke/raw_data/ISOT_Fake_News_Dataset/ISOT_True.csv
# LOCAL_PATH=/Users/julienseguy/code/EddyEdzwan/StopFAIke/raw_data/politifact_scrap.csv
LOCAL_PATH=/Users/julienseguy/code/EddyEdzwan/StopFAIke/raw_data/poynter_final_condensed.csv

# project id
PROJECT_ID=le-wagon-data-bootcamp-321006

# bucket name
BUCKET_NAME=wagon-data-615-seguy

# bucket directory in which to store the uploaded file (we choose to name this data as a convention)
BUCKET_FOLDER=data

# name for the uploaded file inside the bucket folder (here we choose to keep the name of the uploaded file)
# BUCKET_FILE_NAME=another_file_name_if_I_so_desire.csv
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

REGION=europe-west1

set_project:
	-@gcloud config set project ${PROJECT_ID}

create_bucket:
	-@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_data:
	-@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

##### Training  - - - - - - - - - - - - - - - - - - - - - -
# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = trainings

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -
##### Machine configuration - - - - - - - - - - - - - - - -
REGION=us-central1
PYTHON_VERSION=3.7
RUNTIME_VERSION=2.5

##### Package params  - - - - - - - - - - - - - - - - - - -
PACKAGE_NAME=StopFAIke
FILENAME=trainer

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -
JOB_NAME=StopFAIke_training_$(shell date +'%Y%m%d_%H%M%S')


run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--scale-tier=BASIC_TPU \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs



# gcloud ai-platform jobs submit training tpu_mnist_1 \
#   --staging-bucket=gs://BUCKET_NAME \
#   --package-path=official \
#   --module-name=official.vision.image_classification.mnist_main \
#   --runtime-version=2.5 \
#   --python-version=3.7 \
#   --scale-tier=BASIC_TPU \
#   --region=us-central1 \
#   -- \
#   --distribution_strategy=tpu \
#   --data_dir=gs://tfds-data/datasets \
#   --model_dir=gs://BUCKET_NAME/tpu_mnist_1_output
