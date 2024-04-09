import logging
import time
from pm2 import PM2
import git

PM2_NAMES = ["generation", "validation", "miner", "validator", "updater"]
UPDATE_CHECK_INTERVAL = 15  # minutes
pm = PM2()
repo = git.Repo()


def repository_needs_update():
    """
    lkjlj
    """
    remote = repo.remotes.origin
    remote.fetch()
    current_branch = repo.active_branch
    commits_behind = repo.iter_commits(f"{current_branch.name}..{remote.name}/{current_branch.name}")

    if sum(1 for _ in commits_behind) == 0:
        logging.info("Repository is up to date.")
        return True
    else:
        logging.info("Repository is not up to date.")
        return False


def update_repository() -> None:
    """
    This
    """

    try:
        # subprocess.run(split("git pull --rebase --autostash"), check=True, cwd=constants.ROOT_DIR)
        origin = repo.remote(name="origin")
        origin.pull(rebase=True)
        logging.info("")
    except:
        logging.error("Failed to pull, reverting")
        # subprocess.run(split("git rebase --abort"), check=True, cwd=constants.ROOT_DIR)


def restart_processes():
    """
    bla
    """
    processes = pm.list()
    for name in PM2_NAMES:
        if name in processes:
            pm.restart(name=name)


def check_for_updates():
    """
    lsdkjfslkjf
    """
    logging.info("Checking for updates...")
    if repository_needs_update():
        update_repository()
        restart_processes()


def main():
    while True:
        check_for_updates()
        time.sleep(UPDATE_CHECK_INTERVAL * 60)


if __name__ == "__main__":
    main()
