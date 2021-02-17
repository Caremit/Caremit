import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.stride_tricks import as_strided
from numpy.core.overrides import array_function_dispatch

def _sliding_window_view_dispatcher(x, window_shape, step_shape=None, axis=None, *,
                                    subok=None, writeable=None):
    return (x,)


@array_function_dispatch(_sliding_window_view_dispatcher)
def sliding_window_view(x, window_shape, step_shape=None, axis=None, *,
                        subok=False, writeable=False):
    """ like the new numpy native function https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html 
    
    In addition, this method supports a `step_shape` parameter, with which you can move windows by steps other than 1, and also in multiple dimensions.
    HOWEVER CAUTION needs to be taken, as this `step_shape` parameter currently only works well with `axis=None`.
    """
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    step_shape = (tuple(step_shape)
                  if np.iterable(step_shape)
                  else (step_shape,))
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')
    if step_shape is None:
        step_shape = (1,)*len(window_shape)
        
    for ax, dim, step in zip(axis, window_shape, step_shape):
        if (x.shape[ax] - dim) % step != 0:
            raise ValueError("`step_shape` needs to be a dividor of the respective dimension size, minus the window size.")
    
    # TODO this probably does not generalize to arbitrary axis as input... would be nice to make a pull-request to numpy with a full version
    out_strides = tuple(stride*step for stride, step in zip(x.strides, step_shape)) + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim, step in zip(axis, window_shape, step_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] = ((x.shape[ax] - dim) // step) + 1
    
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape,
                      subok=subok, writeable=writeable)