import sys
import os
import torch


class Setup(object):
    def setup(self):
        raise NotImplementedError()

    def shutdown(self):
        raise NotImplementedError()


class FileSetup(object):
    path = None

    def shutdown(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            pass


class EvalModeForLoadedModule(FileSetup):
    path = 'dropout_model.pt'

    def setup(self):
        class Model(torch.jit.ScriptModule):
            def __init__(self):
                super(Model, self).__init__()
                self.dropout = torch.nn.Dropout(0.1)

            @torch.jit.script_method
            def forward(self, x):
                x = self.dropout(x)
                return x

        model = Model()
        model = model.train()
        model.save(self.path)


class SerializationInterop(FileSetup):
    path = 'ivalue.pt'

    def setup(self):
        ones = torch.ones(2, 2)
        twos = torch.ones(3, 5) * 2

        value = (ones, twos)

        torch.save(value, self.path, _use_new_zipfile_serialization=True)


# See testTorchSaveError in test/cpp/jit/tests.h for usage
class TorchSaveError(FileSetup):
    path = 'eager_value.pt'

    def setup(self):
        ones = torch.ones(2, 2)
        twos = torch.ones(3, 5) * 2

        value = (ones, twos)

        torch.save(value, self.path, _use_new_zipfile_serialization=False)

class TorchSaveJitStream_CUDA(FileSetup):
    path = 'saved_stream_model.pt'

    def setup(self):
        if not torch.cuda.is_available():
            return

        class Model(torch.nn.Module):
            def forward(self):
                device_index = torch.cuda._current_device()
                s = torch.jit.cuda.Stream(device_index, 0)
                a = torch.rand(3, 4, device="cuda")
                b = torch.rand(3, 4, device="cuda")

                with torch.jit.cuda.stream(s):
                    is_stream_s = torch.cuda.current_stream(s.device_index()).id() == s.id()
                    c = torch.cat((a, b), 0).to("cuda")
                s.synchronize()
                return is_stream_s, a, b, c

        model = Model()

        # Script the model and save
        script_model = torch.jit.script(model)
        torch.jit.save(script_model, self.path)


tests = [
    EvalModeForLoadedModule(),
    SerializationInterop(),
    TorchSaveError(),
    TorchSaveJitStream_CUDA()
]

def setup():
    for test in tests:
        test.setup()


def shutdown():
    for test in tests:
        test.shutdown()


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "setup":
        setup()
    elif command == "shutdown":
        shutdown()
