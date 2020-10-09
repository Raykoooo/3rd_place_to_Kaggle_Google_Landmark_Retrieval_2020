import torch


class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.data_dict = next(self.loader)
        except StopIteration:
            self.data_dict = None
            return
        with torch.cuda.stream(self.stream):
            for key in self.data_dict:
                if isinstance(self.data_dict[key], torch.Tensor):
                    self.data_dict[key] = self.data_dict[key].cuda(non_blocking=True)

    def __iter__(self):
        while(self.data_dict):
            torch.cuda.current_stream().wait_stream(self.stream)
            _data_dict = self.data_dict
            self.preload()
            yield _data_dict
