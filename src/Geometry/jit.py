from torch.utils.cpp_extension import load
tet_utils = load(
    'tet_utils', ['tet_utils.cpp'], verbose=True)
#help(tet_utils)
