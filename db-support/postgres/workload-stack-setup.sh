#!/bin/bash

set -e  # exit on error

DB_NAME=stack
WD=$(pwd)
TARGET_DIR=$WD/../stack_data
PG_CONN="-U $(whoami)"
FORCE_CREATION="false"
SKIP_EXTENSIONS="false"
SKIP_VACUUM="false"
CLEANUP="false"

show_help() {
    RET=$1
    echo "Usage: $0 <options>"
    echo "Allowed options:"
    echo -e "-d | --dir <directory>\tspecifies the directory to store/load the Stack data files, defaults to '../stack_data'"
    echo -e "-f | --force\t\tdelete existing instance of the database if necessary"
    echo -e "-t | --target <db name>\tname of the Stack database, defaults to 'stack'"
    echo -e "-j | --jobs <count>\tconfigure the number of worker processes to load the database. Defaults to 1/2 of CPU cores"
    echo -e "--pg-conn <connection>\tconnection string to the PostgreSQL server (server, port, user) if required, e.g. '-h localhost -U admin'"
    echo -e "--no-fkeys\t\tdoes not load foreign key indexes to the database (includes both foreign key constraints as well as the actual indexes)"
    echo -e "--no-ext\t\tdoes not install any extensions"
    echo -e "--no-vacuum\t\tdoes not run VACUUM ANALYZE after the import"
    echo -e "--cleanup\t\tremove the data files after import"
    exit $RET
}

attempt_pg_ext_install() {
    EXTENSION="$1"
    AVAILABLE_EXTS="$(psql $PG_CONN $DB_NAME -t -c "SELECT name FROM pg_available_extensions" | grep "$EXTENSION" || true)"
    if [ -z "$AVAILABLE_EXTS" ] ; then
        echo ".. Extension $EXTENSION not available, skipping"
        return
    fi
    psql $PG_CONN "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS $EXTENSION;"
}

while [ $# -gt 0 ] ; do
    case $1 in
        -d|--dir)
            TARGET_DIR=$2
            shift
            shift
            ;;
        -j|--jobs)
            CORES=$2
            shift
            shift
            ;;
        -t|--target)
            DB_NAME=$2
            shift
            shift
            ;;
        -f|--force)
            FORCE_CREATION="true"
            shift
            ;;
        --pg-conn)
            PG_CONN=$2
            shift
            shift
            ;;

        --no-ext)
            SKIP_EXTENSIONS="true"
            shift
            ;;
        --no-vacuum)
            SKIP_VACUUM="true"
            shift
            ;;
        --cleanup)
            CLEANUP="true"
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

EXISTING_DBS="$(psql $PG_CONN -l | grep " $DB_NAME " || true)"

if [ ! -z "$EXISTING_DBS" ] && [ $FORCE_CREATION = "false" ] ; then
    echo ".. Stack database exists, doing nothing"
    exit 0
fi

if [ ! -z "$EXISTING_DBS" ] ; then
    dropdb $PG_CONN "$DB_NAME"
fi

echo ".. Initializing Stack data dir at $TARGET_DIR"
mkdir -p "$TARGET_DIR"

if [ -f "$TARGET_DIR/stack_dump" ] ;  then
    echo ".. Re-using existing Stack dump"
else
    echo ".. Downloading Stack database dump"
    curl --location "https://www.dropbox.com/s/55bxfhilcu19i33/so_pg13?dl=1" --output "$TARGET_DIR/stack_dump"
fi

echo ".. Creating Stack database"
createdb $PG_CONN "$DB_NAME"

if [ $SKIP_EXTENSIONS == "false" ] ; then
    attempt_pg_ext_install "pg_buffercache"
    attempt_pg_ext_install "pg_prewarm"
    attempt_pg_ext_install "pg_cooldown"
    attempt_pg_ext_install "pg_hint_plan"
else
    echo ".. Skipping extension generation"
fi

OS_TYPE="$(uname)"
if [[ "$OS_TYPE" = "Darwin" ]]; then
    NCORES="$(sysctl -n hw.logicalcpu)"
elif [[ "$OS_TYPE" = "Linux" ]]; then
    NCORES="$(nproc)"
else
    NCORES=1
fi
MAX_CONN="$(psql -t -c 'show max_connections' | tr -d '[:space:]')"
NJOBS="$((NCORES < MAX_CONN ? NCORES : MAX_CONN - 1))"

echo ".. Loading Stack dump"
pg_restore $PG_CONN -O -x --exit-on-error --jobs="$NJOBS" --dbname="$DB_NAME" "$TARGET_DIR/stack_dump"

if [ $SKIP_VACUUM == "false" ] ; then
    echo ".. Vacuuming database"
    psql $PG_CONN "$DB_NAME" -c "VACUUM ANALYZE;"
else
    echo ".. Skipping vacuuming"
fi

if [ $CLEANUP == "true" ] ; then
    echo ".. Removing data files"
    rm -rf "$TARGET_DIR"
fi

echo ".. Done"
