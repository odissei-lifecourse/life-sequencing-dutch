#!/bin/bash
# This is how the venv on regular snellius should be activated. For OSSC, change the ENV_NAME if needed.

declare ENV_NAME=".venv"

source requirements/snel_modules_2023.sh
source ${ENV_NAME}/bin/activate
