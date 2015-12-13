from invoke import run, task 

@task
def clean():
    run("rm *.pyc")
