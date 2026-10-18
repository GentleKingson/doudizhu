"""Colab 0.7.4 workaround: attach automation to the cached kernel and await readiness.

Run with the installed colab CLI's Python, followed by the usual colab arguments.
"""
from colab_cli.commands import automation, execution
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime


class AttachedRuntime(ColabRuntime):
    def __init__(self, url, token, **kwargs):
        if kwargs.get('session_name'):
            session=state.get_session(kwargs['session_name'])
            if not session.kernel_id:
                raise RuntimeError('Run colab exec once to establish this session kernel before automation')
            kwargs.setdefault('kernel_id',session.kernel_id)
            kwargs.setdefault('session_id',session.session_id)
        super().__init__(url,token,**kwargs)

    def execute_code(self,*args,**kwargs):
        self.kernel_client._manager.client.wait_for_ready(timeout=30)
        if self.kernel_id and self.kernel_client.id!=self.kernel_id:
            raise RuntimeError('Colab attached a different kernel')
        return super().execute_code(*args,**kwargs)


automation.ColabRuntime=AttachedRuntime
execution.ColabRuntime=AttachedRuntime
if __name__=='__main__':
    from colab_cli.cli import app
    app()
