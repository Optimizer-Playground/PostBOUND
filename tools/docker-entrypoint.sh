#!/bin/bash

if [ -z "$SETUP_POSTGRES" ] ; then
    echo "[setup] SETUP_POSTGRES is not set, defaulting to true"
    SETUP_POSTGRES="true"
fi

if [ -z "$PG_DISK_TYPE" -a "$OPTIMIZE_PG_CONFIG" = "true" ] ; then
    echo "[setup] PG_DISK_TYPE is not set, but PG server should be optimized. Assuming SSD."
    PG_DISK_TYPE="SSD"
fi

if [ -z "$PGVER" ] ; then
    echo "[setup] PGVER is not set, using default version 18"
    PGVER="18"
fi

if [ "$USE_PGLAB" = "true" ] ; then
    PGPATH=/pg_lab
else
    PGPATH=/postbound/db-support/postgres
fi

if [ -z "$SETUP_DUCKDB" ] ; then
    echo "[setup] SETUP_DUCKDB is not set, defaulting to false"
    SETUP_DUCKDB="false"
fi

if [ -z "$(ls /postbound)" ] ; then

    echo "[setup] No files found in /postbound, starting initial setup"

    sudo chown -R $USERNAME:$USERNAME /postbound
    sudo chmod -R 755 /postbound

    sudo chown -R $USERNAME:$USERNAME /pg_lab
    sudo chmod -R 755 /pg_lab

    git clone --depth 1 --branch=main https://github.com/Optimizer-Playground/PostBOUND /postbound

    # Setup local Postgres or pg_lab
    if [ "$SETUP_POSTGRES" = "true" -a "$USE_PGLAB" = "true" ] ; then
        if [ -z "$(ls /pg_lab)" ] ; then
            echo "[setup] Building pg_lab"
            git clone --depth 1 --branch=main https://github.com/rbergm/pg_lab /pg_lab
            cd /pg_lab && ./postgres-setup.sh --pg-ver "$PGVER" --remote-password "postbound" --stop
            . ./postgres-start.sh
        else
            echo "[setup] Reusing existing pg_lab installation"
            cd /pg_lab && . ./postgres-start.sh
        fi
    elif [ "$SETUP_POSTGRES" = "true" ] ; then
        if [ -z "$(ls /postbound/db-support/postgres/postgres-server)" ] ; then
            echo "[setup] Building vanilla Postgres server"
            cd /postbound/db-support/postgres && ./postgres-setup.sh --pg-ver "$PGVER" --remote-password "postbound" --stop
            . ./postgres-start.sh
        else
            echo "[setup] Reusing existing Postgres installation"
            cd /postbound/db-support/postgres && . ./postgres-start.sh
        fi
    fi

    cd /postbound/db-support/postgres
    if [ "$OPTIMIZE_PG_CONFIG" = "true" ] ; then
        python3 postgres-config-generator.py --out pg-conf.sql --disk-type "$PG_DISK_TYPE" "$PGDATA"
        psql -f pg-conf.sql
    fi

    # Setup databases
    if [ "$SETUP_POSTGRES" = "true" -a "$SETUP_IMDB" = "true" ] ; then
        echo "[setup] Setting up IMDB benchmark for Postgres"
        cd /postbound/db-support/postgres
        ./workload-job-setup.sh
        ./postgres-psycopg-setup.sh job imdb
    fi
    if [ "$SETUP_POSTGRES" = "true" -a "$SETUP_STATS" = "true" ] ; then
        echo "[setup] Setting up Stats benchmark for Postgres"
        cd /postbound/db-support/postgres
        ./workload-stats-setup.sh
        ./postgres-psycopg-setup.sh stats stats
    fi
    if [ "$SETUP_POSTGRES" = "true" -a "$SETUP_STACK" = "true" ] ; then
        echo "[setup] Setting up Stack benchmark for Postgres"
        cd /postbound/db-support/postgres
        ./workload-stack-setup.sh
        ./postgres-psycopg-setup.sh stack stack

        cd /postbound/workloads/Stack-Queries
        ./setup.sh
        cd /postbound/db-support/postgres
    fi
    mv  .psycopg_connection_* /postbound
    cd /postbound
    if [ $(ls -a /postbound | grep .psycopg_connection | wc -l ) -eq 1 ]; then
        ln -s $(ls -a /postbound | grep .psycopg_connection) .psycopg_connection
    fi

    # Install PostBOUND package
    cd /postbound && tools/setup-py-venv.sh --venv /postbound/pb-venv
    source /postbound/pb-venv/bin/activate

    # Setup DuckDB
    # This needs to happen _after_ the PostBOUND package is installed, because we install the DuckDB Python package
    # in the same virtual environment
    if [ "$SETUP_DUCKDB" = "true" ] ; then
        echo "[setup] Setting up DuckDB"
        cd /postbound/db-support/duckdb
        ./duckdb-setup.sh --venv /postbound/pb-venv
    fi

    if [ "$SETUP_DUCKDB" = "true" -a "$SETUP_IMDB" = "true" ] ; then
        echo "[setup] Setting up IMDB benchmark for DuckDB"
        cd /postbound/db-support/duckdb
        ./workload-imdb-setup.sh
    fi
    if [ "$SETUP_DUCKDB" = "true" -a "$SETUP_STATS" = "true" ] ; then
        echo "[setup] Setting up Stats benchmark for DuckDB"
        cd /postbound/db-support/duckdb
        ./workload-stats-setup.sh
    fi
    mv /postbound/db-support/duckdb/*.duckdb /postbound

    # Clean up raw database files
    echo "[setup] Cleaning up"
    rm -rf /postbound/db-support/imdb_data
    rm -rf /postbound/db-support/stats_data
    rm -rf /postbound/db-support/stack_data

    # User config
    if [ "$SETUP_POSTGRES" = "true" -a "$USE_PGLAB" = "true" ] ; then
        echo "cd /pg_lab/" >> /home/$USERNAME/.bashrc
        echo "source ./postgres-load-env.sh" >> /home/$USERNAME/.bashrc
    elif [ "$SETUP_POSTGRES" = "true" ] ; then
        echo "cd /postbound/db-support/postgres/" >> /home/$USERNAME/.bashrc
        echo "source ./postgres-load-env.sh" >> /home/$USERNAME/.bashrc
    fi

    # Currently the DuckDB/quacklab setup does not create a duckdb executable.
    # Re-enable this once we found a way to build both the Python package and the CLI tool in one step.
    # if [ "$SETUP_DUCKDB" = "true" ] ; then
    #     echo "cd /postbound/db-support/duckdb" >> /home/$USERNAME/.bashrc
    #     echo "source ./duckdb-env.sh" >> /home/$USERNAME/.bashrc
    # fi

    echo "cd /postbound" >> /home/$USERNAME/.bashrc
    echo "source /postbound/pb-venv/bin/activate" >> /home/$USERNAME/.bashrc


    echo "[setup] Installation complete. You can now start using PostBOUND."

else

    echo "Starting Postgres server"
    cd $PGPATH
    . ./postgres-start.sh

fi



tail -f /dev/null
