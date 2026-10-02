#!/usr/bin/env python3

import datetime
import os
import pathlib
import subprocess
import sys
import textwrap


def print_err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


MysqlSafePath = "/var/lib/mysql-files"

MysqlQueryCmd = '''mysql -s -r -N -uroot imdb -e "{query}"'''
# -s + -r: silent and raw (= tab-separated output), -N: no column names

MysqlColumnsQuery = textwrap.dedent("""
                                    SELECT column_name, is_nullable
                                    FROM information_schema.columns
                                    WHERE table_schema = 'imdb' AND table_name = '{table}'
                                    ORDER BY ordinal_position""")


MysqlImportCmd = '''mysql --local-infile -uroot imdb -e "{cmd}"'''
MysqlImportQuery = textwrap.dedent("""
                                 LOAD DATA
                                 INFILE '/var/lib/mysql-files/{filename}'
                                 INTO TABLE {table}
                                 FIELDS TERMINATED BY '|'
                                 ENCLOSED BY '\b'
                                 ({virtual_cols})
                                 SET
                                 {col_wrappings}
                                 """)


def main():
    print("Setting up IMDB instance", flush=True)
    res_code = os.system(""" mysql -uroot -e "CREATE DATABASE imdb" """)
    if res_code:
        print_err("Could not create IMDB database")
        sys.exit(1)

    res_code = os.system(""" mysql -uroot imdb < create.sql """)
    if res_code:
        print_err("Could not load IMDB schema")
        sys.exit(1)

    res_code = os.system(""" mysql -uroot imdb < fkindexes.sql """)
    if res_code:
        print_err("Could not create IMDB foreign key indexes")
        sys.exit(1)

    print("Starting data import", flush=True)
    os.chdir(MysqlSafePath)
    for file in pathlib.Path(".").glob("*.csv"):
        print(f"[{datetime.datetime.now()}] Preparing file {file}", flush=True)
        os.system(f"sed -i 's/\x08//g' {file}")

        table_name = file.stem
        print(f"[{datetime.datetime.now()}] Now importing table {table_name}", flush=True)

        columns_query = MysqlColumnsQuery.format(table=table_name)
        query_cmd = MysqlQueryCmd.format(query=columns_query)
        raw_table_info = subprocess.check_output(query_cmd, shell=True, text=True)

        column_info = {}
        for raw_column_info in raw_table_info.split("\n"):
            if not raw_column_info.strip():
                continue
            col_name, nullable_col = raw_column_info.split()
            column_info[col_name] = nullable_col == "YES"

        virtual_cols = ", ".join(f"@v{col_name}" for col_name in column_info)
        col_wrapping = ",\n".join(
            f"{col_name} = NULLIF(@v{col_name}, '')" if nullable_col else f"{col_name} = @v{col_name}"
            for col_name, nullable_col in column_info.items()
        )

        import_query = MysqlImportQuery.format(
            filename=file, table=table_name, virtual_cols=virtual_cols, col_wrappings=col_wrapping
        )
        print("... Using import query", import_query, flush=True)
        import_cmd = MysqlImportCmd.format(cmd=import_query)
        proc_res = subprocess.run(import_cmd, shell=True)
        if proc_res.returncode:
            print_err(f"Could not import data for table {table_name}")
            sys.exit(1)


if __name__ == "__main__":
    main()
