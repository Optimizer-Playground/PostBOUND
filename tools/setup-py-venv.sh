#!/bin/bash

# Installs PostBOUND and its dependencies via uv, for *using* PostBOUND (this is what the
# Docker entrypoint calls). The dev tools are deliberately left out.
#
# If you are working ON PostBOUND rather than with it, don't use this script -- just run
# `uv sync` and `uv run pre-commit install`. See CONTRIBUTING.md.

set -e

WD=$PWD
TARGET_DIR=""
EXTRAS="--all-extras"
BUILD_DOC="false"
GIT_PULL="true"

show_help() {
  RET=$1
  echo -e "Usage: $0 <options>"
  echo -e ""
  echo -e "Installs PostBOUND into a Python virtual environment using uv. This script is assumed to be run from the"
  echo -e "root of the PostBOUND repository, i.e. as tools/setup-py-venv.sh."
  echo -e "If PostBOUND is already installed, it will be upgraded."
  echo -e ""
  echo -e "uv is required. If it is not installed, get it from https://docs.astral.sh/uv/ or run:"
  echo -e "\tcurl -LsSf https://astral.sh/uv/install.sh | sh"
  echo -e ""
  echo -e "Allowed options:"
  echo -e "\n--venv <dir>"
  echo -e "\tPath to the virtual environment where PostBOUND will be installed. Defaults to ./.venv, which is where"
  echo -e "\tuv run and the pre-commit hooks look for it. Only override this if you know you need to."
  echo -e "\n--features <features>"
  echo -e "\tOptional extras to install with PostBOUND, as a comma-separated list. Supported extras are 'vis', 'duckdb'"
  echo -e "\tand 'mysql'. 'all' (the default) installs all of them, 'minimal' installs only the core package."
  echo -e "\n--include-doc"
  echo -e "\tAlso build the documentation."
  echo -e "\n--skip-pull"
  echo -e "\tDon't pull the latest version of the repository before building. Notice that a pull will not update this script"
  echo -e "\twhile it is running. If there should be any issues with the setup script, please pull the latest version "
  echo -e "\tmanually and try again."
  echo -e "\n--help"
  echo -e "\tShow this help message."
  exit $RET
}

while [ $# -gt 0 ] ; do
  case $1 in
    --venv)
      TARGET_DIR="$2"
      shift
      shift
      ;;
    --features)
      case $2 in
        all)
          EXTRAS="--all-extras"
          ;;
        minimal)
          EXTRAS="--no-extra vis --no-extra duckdb --no-extra mysql"
          ;;
        *)
          EXTRAS=""
          for extra in ${2//,/ } ; do
            EXTRAS="$EXTRAS --extra $extra"
          done
          ;;
      esac
      shift
      shift
      ;;
    --include-doc)
      BUILD_DOC="true"
      shift
      ;;
    --skip-pull)
      GIT_PULL="false"
      shift
      ;;
    --help)
      show_help 0
      ;;
    *)
      show_help 1
      ;;
  esac
done

if ! command -v uv > /dev/null 2>&1 ; then
  echo "!! uv is required but was not found on PATH."
  echo "!! Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [ "$GIT_PULL" == "true" ] ; then
  echo ".. Checking for latest version of PostBOUND"
  git pull
fi

if [ -n "$TARGET_DIR" ] ; then
  echo ".. Installing into virtual environment $TARGET_DIR"
  export UV_PROJECT_ENVIRONMENT="$TARGET_DIR"
else
  echo ".. Installing into the default virtual environment .venv"
fi

# uv provisions the interpreter named in .python-version itself, so there is no need to
# check the system Python version or bootstrap one.
# --no-dev keeps the linting/notebook tooling out of a user installation. It also makes
# --features meaningful: the dev group depends on postbound[duckdb,mysql,vis], so with the
# dev group enabled every extra would be pulled in regardless of what was requested here.
echo ".. Installing PostBOUND and its dependencies"
if [ "$BUILD_DOC" == "true" ] ; then
  uv sync --no-dev $EXTRAS --group doc
else
  uv sync --no-dev $EXTRAS
fi

if [ "$BUILD_DOC" == "true" ] ; then
  echo ".. Building documentation"
  cd "$WD/docs"
  uv run --group doc sphinx-apidoc --force \
                                   --ext-autodoc \
                                   --maxdepth 4 \
                                   --module-first \
                                   -o source/generated \
                                   ../postbound
  uv run --group doc sphinx-build -M html source build
  cd "$WD"
fi

VENV_PATH="${TARGET_DIR:-$WD/.venv}"
echo ".. Done. Run commands with 'uv run <cmd>', or activate the venv as '. $VENV_PATH/bin/activate'."
