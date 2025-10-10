import LabelBench.model.model_impl
from LabelBench.skeleton.model_skeleton import model_fns


def get_model_fn(name):
    fn = model_fns[name]
    return fn
