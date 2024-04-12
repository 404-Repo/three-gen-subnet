import argparse
from datetime import datetime, timedelta
import logging
from shlex import split
import subprocess
import time

import git
from pm2 import PM2


PM2_NAMES = ["generation", "validation", "miner", "validator", "updater"]
DEFAULT_UPDATE_CHECK_INTERVAL = 15  # minutes
DEFAULT_UPDATE_DELAY = 30  # minutes

pm = PM2()
repo = git.Repo(search_parent_directories=True)
update_check_interval = DEFAULT_UPDATE_CHECK_INTERVAL
update_delay = DEFAULT_UPDATE_DELAY


def needs_update():
    """
    Check if remote repo has new commits
    """
    current_datetime = datetime.now().astimezone()
    remote = repo.remotes.origin
    remote.fetch()
    current_branch = repo.active_branch
    commits_behind = list(repo.iter_commits(f"{current_branch.name}..{remote.name}/{current_branch.name}"))

    if commits_behind and current_datetime > commits_behind[-1].committed_datetime + timedelta(minutes=update_delay):
        logging.info("Repository needs update.")
        return True

    logging.info("Repository is up to date.")
    return False


def update_repository() -> None:
    """
    Pull from remote repository
    """

    try:
        subprocess.run(split("git pull --rebase --autostash"), check=True)
    except:
        logging.error("Failed to pull, reverting")
        subprocess.run(split("git rebase --abort"), check=True)


def restart_processes():
    """
    Restart active three-gen processes
    """
    processes = pm.list()
    for p in processes:
        if p.name in PM2_NAMES:
            logging.info(f"Restarting process {p.name}")
            pm.restart(name=p.name)


def check_for_updates():
    """
    Check for updates and
    """
    logging.info("Checking for updates...")
    if needs_update():
        update_repository()
        restart_processes()


def main(args):
    while True:
        try:
            check_for_updates()
            time.sleep(update_check_interval * 60)
        except Exception as e:
            logging.exception("An error occurred:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=int, default=DEFAULT_UPDATE_DELAY, help="Min time since commit in minutes")
    parser.add_argument(
        "--check", type=int, default=DEFAULT_UPDATE_CHECK_INTERVAL, help="Update check interval in minutes"
    )
    args = parser.parse_args()

    update_check_interval = args.check
    update_delay = args.delay

    logging.basicConfig(level=logging.INFO)

    main(args)
