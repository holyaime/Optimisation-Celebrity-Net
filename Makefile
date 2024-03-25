#Makefile


.PHONY: quality test security-check

poetry_install:
	poetry install

quality: poetry_install ##for checking code quality
	poetry run pre-commit run check-yaml
	poetry run pre-commit run trailing-whitespace-fixer
	poetry run pre-commit run end-of-file-fixer
	poetry run pre-commit run check-docstring-first
	poetry run pre-commit run check-merge-conflict
	poetry run pre-commit run fix-encoding-pragma
	poetry run pre-commit run no-commit-to-branch
	poetry run pre-commit run check-added-large-files
	poetry run pre-commit run code-formater
	poetry run pre-commit run sort-imports
	poetry run pre-commit run linter


test: poetry_install ##running test with pytest
	poetry run pytest -v


security: poetry_install ##for security checking
	poetry run pre-commit run bandit 
	poetry run pre-commit run semgrep