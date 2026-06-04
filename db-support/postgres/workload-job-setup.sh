#!/bin/bash

set -e  # exit on error

WD="$(pwd)"
DB_NAME="imdb"
FORCE_CREATION="false"
TARGET_DIR="../imdb_data"
PG_CONN="-U $(whoami)"
SKIP_EXTENSIONS="false"
SKIP_VACUUM="false"
CLEANUP="false"

show_help() {
    RET="$1"
    echo "Usage: $0 <options>"
    echo "Allowed options:"
    echo -e "-d | --dir <directory>\tspecifies the directory to store/load the IMDB data files, defaults to '$TARGET_DIR'"
    echo -e "-f | --force\t\tdelete existing instance of the database if necessary"
    echo -e "-t | --target <db name>\tname of the IMDB database, defaults to 'imdb'"
    echo -e "--pg-conn <connection>\tconnection string to the PostgreSQL server (server, port, user) if required, e.g. '-h localhost -U admin'"
    echo -e "--no-ext\t\tdoes not install any extensions"
    echo -e "--no-vacuum\t\tdoes not run VACUUM ANALYZE after the import"
    echo -e "--cleanup\t\tremove the data files after import"
    exit "$RET"
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
            TARGET_DIR="$2"
            shift
            shift
            ;;
        -f|--force)
            FORCE_CREATION="true"
            shift
            ;;
        -t|--target)
            DB_NAME="$2"
            shift
            shift
            ;;
        --no-fkeys)
            echo "--no-fkeys is obsolete. Foreign keys are always created." >&2
            shift
            ;;
        --pg-conn)
            PG_CONN="$2"
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

echo ".. Working directory is $WD"

if [ ! -z "$EXISTING_DBS" ] && [ "$FORCE_CREATION" = "false" ] ; then
    echo ".. IMDB exists, doing nothing"
    exit 0
fi

if [ ! -z "$EXISTING_DBS" ] ; then
    dropdb $PG_CONN "$DB_NAME"
fi

echo ".. IMDB source directory is $TARGET_DIR"

SRC_FILE="$TARGET_DIR/imdb_dump.pg"
if [ -f "$SRC_FILE" ] ; then
    echo ".. Re-using existing IMDB input data"
else
    echo ".. IMDB source directory does not exist, re-creating"
    echo ".. Fetching IMDB data"
    mkdir -p "$TARGET_DIR"
    curl -k -L -o "$SRC_FILE" "https://zenodo.org/records/19205561/files/imdb_dump.pg?download=1"
fi

echo ".. Creating IMDB database"
createdb $PG_CONN $DB_NAME

if [ $SKIP_EXTENSIONS == "false" ] ; then
    attempt_pg_ext_install "pg_buffercache"
    attempt_pg_ext_install "pg_prewarm"
    attempt_pg_ext_install "pg_hint_plan"
else
    echo ".. Skipping extension generation"
fi

echo ".. Loading IMDB dump"
echo "   This might raise some errors due to missing or existing extensions. It is generally safe to ignore these."
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
pg_restore $PG_CONN -n public -O -d "$DB_NAME" -j "$NJOBS" "$SRC_FILE"

if [ $SKIP_VACUUM == "false" ] ; then
    echo ".. Vacuuming database"
    psql $PG_CONN "$DB_NAME" -c "VACUUM ANALYZE;"
else
    echo ".. Skipping vacuuming"
fi

if [ $CLEANUP == "true" ] ; then
    echo ".. Removing data files"
    rm -rf "$SRC_FILE"
fi

echo ".. Done"
