class Module: # incomplete
    def __init__(self):
        self._modules = {}

    def add_module(
        self,
        name: str,
        module
    ):
        self._modules[name] = module

    def parameters(self):
        for m in self._modules.values():
            yield from m.parameters()

class ModuleList(Module):
    def __init__(
        self,
        modules=None
    ):
        super().__init__()
        self._list = []

        if modules is not None:
            for m in modules: self.append(m)

    def append(
        self,
        module
    ):
        idx = str(len(self._list))
        self.add_module(idx, module)
        self._list.append(module)

    def __getitem__(self, idx):
        return self._list[idx]
    
    def __len__(self):
        return len(self._list)
    
    def __iter__(self):
        return iter(self._list)

class ModuleDict(Module):
    def __init__(
        self,
        modules=None
    ):
        super().__init__()
        self._dict = {}
        if modules is not None:
            for name, m in modules.items(): self[name] = m

    def __setitem__(
        self,
        key,
        module
    ):
        self.add_module(key, module)
        self._dict[key] = module

    def __getitem__(self, key):
        return self._dict[key]
    
    def __iter__(self):
        return iter(self._dict)
    
    def items(self):
        return self._dict.items()
    
    def __len__(self):
        return len(self._dict)
