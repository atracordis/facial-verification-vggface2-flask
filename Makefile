VENV_ACTIVATE_FILE = ./activate_venv

venv:  ## Create virtualenv
	python -m venv venv
	. $(VENV_ACTIVATE_FILE) && pip install -r requirements-dev.txt
	. $(VENV_ACTIVATE_FILE) && pip install git+https://github.com/atracordis/keras-vggface.git