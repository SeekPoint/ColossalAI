import click
from pydebug import gd, infoTensor
from .check import check
from .launcher import run


class Arguments:
    def __init__(self, arg_dict):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        for k, v in arg_dict.items():
            self.__dict__[k] = v
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')


@click.group()
def cli():
    pass


cli.add_command(run)
cli.add_command(check)

if __name__ == "__main__":
    cli()
