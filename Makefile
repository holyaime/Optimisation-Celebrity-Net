.ONESHELL:

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


test: poetry_install ##running test with mypy and pytest
	poetry run pre-commit run my_py
	poetry run pre-commit run test


security: poetry_install ##for security checking
	poetry run pre-commit run bandit 
	poetry run pre-commit run semgrep

