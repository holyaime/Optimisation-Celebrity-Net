#Makefile

.PHONY: quality test security-check

quality: ##for checking code quality
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


test: ##running test with pytest
	poetry run pytest -v


security-check: ##for security checking
	poetry run pre-commit run security
	poetry run pre-commit run semgrep