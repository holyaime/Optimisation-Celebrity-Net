.ONESHELL:
# Variables
# PATH_TO_DATASET := "/home/beranger/Downloads/Rice_Image_Dataset/"
HOST_IP := "0.0.0.0"
FILENAME := "/home/beranger/Téléchargements/celebrity-data/arafat-dj/f2629e10-8374-4cfc-9c22-966e3ee0f188.jpg"
PORT := 5001

.PHONY: quality test security-check dev

poetry_install:
	poetry install

dev: poetry_install ##to install project dependencies and activate pre-commit locally
	poetry run pre-commit install

quality: poetry_install ##for checking code quality
	poetry run pre-commit run check-yaml --all-files
	poetry run pre-commit run trailing-whitespace-fixer --all-files
	poetry run pre-commit run end-of-file-fixer --all-files
	poetry run pre-commit run check-docstring-first --all-files
	poetry run pre-commit run check-merge-conflict --all-files
	poetry run pre-commit run fix-encoding-pragma --all-files
	poetry run pre-commit run no-commit-to-branch --all-files
	poetry run pre-commit run check-added-large-files --all-files
	poetry run pre-commit run code-formater --all-files
	poetry run pre-commit run sort-imports --all-files
	poetry run pre-commit run linter --all-files
	poetry run pre-commit run typing --all-files


test: poetry_install ##running test with mypy and pytest
	poetry run pre-commit run test

security: poetry_install ##for security checking
	poetry run pre-commit run bandit 
	poetry run pre-commit run semgrep

build:
	poetry export -f requirements.txt --only main --without-hashes --without-urls -o requirements.txt
	poetry build
	docker build -t celebritynet --build-arg MODEL="celebritynet.pth" \
	-f celebrity_recognition_ai/app/Dockerfile .
	rm -rf dist
	rm requirements.txt

build_pipeline:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) --build-arg MODEL="celebritynet.pth" \
	-f celebrity_recognition_ai/app/Dockerfile .
	rm -rf dist
	rm requirements.txt

run:
	docker run -d -p 5001:5001 celebritynet
	
predict:
	python celebrity_recognition_ai/app/frontend.py --filename ${FILENAME} --host-ip ${HOST_IP} --port ${PORT}
