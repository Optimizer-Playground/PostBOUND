#!/usr/bin/env python3

from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import textwrap
from datetime import datetime
from typing import Literal, Optional

DiskType = Literal["HDD", "SSD"]


def determine_number_of_cores() -> int:
    """Calculate the number of CPU cores available on the current system."""
    return int(subprocess.check_output(["nproc", "--all"]))


def determine_memory_size_mb() -> int:
    """Calculate the total amount of main memory (in MB) available on the current system."""

    # free output format:
    #                total        used        free      shared  buff/cache   available
    # Mem:           15924        2064       11249         109        2609       13426
    # Swap:           4096           0        4096
    return int(subprocess.check_output(["free", "-m"]).split()[7])


def determine_disk(directory: str) -> str:
    """Determine the disk that the given directory is located on, such as sda or sdb."""

    # based on https://unix.stackexchange.com/questions/11311/how-do-i-find-on-which-physical-device-a-folder-is-located)

    # df output format:
    # Filesystem     1K-blocks      Used Available Use% Mounted on
    # /dev/sda1       12345678    123456  12345678   1% /mnt
    current_disk = subprocess.check_output(f"df {directory} | grep '^/' | cut -d' ' -f1", shell=True, text=True).strip()

    # Resolve logical volumes
    full_path = subprocess.check_output(["realpath", current_disk], text=True).strip().split("/")[-1]
    return full_path


def determine_disk_type(disk_file: Optional[str] = None) -> DiskType:
    """Check, whether a given disk (e.g. sda) is an HDD  or an SSD.

    If no disk is provided, the disk containing the current working directory is used.
    """
    # based on: https://unix.stackexchange.com/questions/65595/how-to-know-if-a-disk-is-an-ssd-or-an-hdd
    disk_file = determine_disk(os.getcwd()) if disk_file is None else disk_file
    if not os.path.exists(f"/sys/block/{disk_file}/queue/rotational"):
        # quick and dirty workaround for Docker, etc. deployments, crop sda1 to sda and try again
        disk_file = disk_file[:-1]

    path = f"/sys/block/{disk_file}/queue/rotational"
    rotational_info = int(subprocess.check_output(["cat", path], text=True))
    return "HDD" if rotational_info == 1 else "SSD"


@dataclasses.dataclass(frozen=True)
class SystemInfo:
    """Wrapper class for the relevant hardware properties."""

    n_cores: int
    memory_mb: int
    disk_type: DiskType

    @staticmethod
    def load(db_directory: str, *, disk_type: Optional[DiskType] = None) -> SystemInfo:
        """Automatically determines the properties of the current system.

        The db_directory is requrired to determine whether the database is located on an HDD or an SSD.
        """
        return SystemInfo(
            n_cores=determine_number_of_cores(),
            memory_mb=determine_memory_size_mb(),
            disk_type=disk_type if disk_type else determine_disk_type(determine_disk(db_directory)),
        )

    @property
    def memory_gb(self) -> int:
        return self.memory_mb // 1024


def generate_pg_config(system_info: SystemInfo) -> dict[str, str]:
    # settings and formulas based on PGTune by le0pard
    # rules extracted from https://github.com/le0pard/pgtune/blob/master/src/features/configuration/configurationSlice.js

    pg_config: dict[str, object] = {}

    max_connections = 40
    pg_config["max_connections"] = max_connections

    pg_config["huge_pages"] = "try" if system_info.memory_gb >= 32 else "off"

    shared_buffers_mb = round(0.25 * system_info.memory_mb)
    pg_config["shared_buffers"] = f"{shared_buffers_mb}MB"

    effective_cache_size_mb = round(0.75 * system_info.memory_mb)
    pg_config["effective_cache_size"] = f"{effective_cache_size_mb}MB"

    maintenance_work_mem_gb = min(2, round(1 / 8 * system_info.memory_gb))
    pg_config["maintenance_work_mem"] = f"{maintenance_work_mem_gb}GB"

    pg_config["min_wal_size"] = "4GB"
    pg_config["max_wal_size"] = "16GB"

    pg_config["checkpoint_completion_target"] = 0.9

    wal_buffers_mb = min(16, round(0.03 * shared_buffers_mb))
    pg_config["wal_buffers"] = f"{wal_buffers_mb}MB"

    pg_config["default_statistics_target"] = 500

    pg_config["random_page_cost"] = 1.1 if system_info.disk_type == "SSD" else 4.0

    pg_config["effective_io_concurrency"] = 200 if system_info.disk_type == "SSD" else 2

    max_parallel_workers_per_gather = round(0.5 * system_info.n_cores)
    pg_config["max_parallel_workers_per_gather"] = max_parallel_workers_per_gather

    pg_config["max_worker_processes"] = system_info.n_cores
    pg_config["max_parallel_workers"] = system_info.n_cores

    maintenance_workers = min(4, round(0.5 * system_info.n_cores))
    pg_config["max_parallel_maintenance_workers"] = maintenance_workers

    work_mem_mb = round(
        ((system_info.memory_mb - shared_buffers_mb) / (3 * max_connections) / max_parallel_workers_per_gather / 2)
    )
    pg_config["work_mem"] = f"{work_mem_mb}MB"

    return {conf_key: str(conf_value) for conf_key, conf_value in pg_config.items()}


def make_config_head(db_directory: str) -> str:
    creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return textwrap.dedent(f"""
                           -- DB configuration generated by postgres-config-generator.py
                           -- Generated on: {creation_time}
                           -- Data directory: '{db_directory}'
                           -- Generator based on rules from PGTune by le0pard (https://pgtune.leopard.in.ua/)

                           """)


def export_pg_config(pg_config: dict[str, str], *, db_directory: str, out_path: str) -> None:
    alter_statements = [f"ALTER SYSTEM SET {conf_key} = '{conf_value}';" for conf_key, conf_value in pg_config.items()]
    config_body = "\n".join(alter_statements) + "\n"
    header = make_config_head(db_directory)

    with open(out_path, "w") as out_file:
        out_file.write(header)
        out_file.write(config_body)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an optimized PostgreSQL configuration file",
        epilog="Generation rules are based on PGTune by le0pard (https://pgtune.leopard.in.ua/)",
    )
    parser.add_argument(
        "db_directory",
        default="postgres-server/data",
        nargs="?",
        help="The Postgres data/ directory containing the database files",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="pg-conf.sql",
        help="The output file for the generated configuration",
    )
    parser.add_argument(
        "--disk-type",
        default="",
        choices=["", "SSD", "HDD", "ssd", "hdd"],
        help="Whether the configuration should be optimized for SSD or HDD. If not provided, the disk type is "
        "determined automatically based on the data/ directory.",
    )

    args = parser.parse_args()

    system_info = SystemInfo.load(args.db_directory, disk_type=args.disk_type.upper())
    pg_config = generate_pg_config(system_info)
    export_pg_config(pg_config, db_directory=args.db_directory, out_path=args.out)


if __name__ == "__main__":
    main()
