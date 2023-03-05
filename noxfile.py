import nox


@nox.session
def lint(session):
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure")


@nox.session(venv_backend="none")
def test(session):
    session.run("pdm", "sync", "-d", "-G", "dev", external=True)
    session.run(
        "pdm",
        "run",
        "pytest",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-report=term",
        "--cov=gym_distractions",
        *session.posargs,
    )
