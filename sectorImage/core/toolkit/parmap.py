import multiprocessing as mp
import time

try:
    # Try to set the start method
    # This is not necessary needed, depending on what you try to parallelize the default 'fork' works fine
    # e.g. OpenCV under OSX needs this
    mp.set_start_method('spawn')
except RuntimeError as e:
    # Pass if it has already been set
    # A RuntimeError is hopefully only raised for this reason;
    # Otherwise this should be handled a bit more carefully
    pass


def fun(f, q_in, q_out, args, kwargs):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x, *args, **kwargs)))


def Parmap(f, X, *args, **kwargs):
    t0 = time.time()
    nprocs = min(mp.cpu_count(), 8)
    q_in = mp.Queue(1)
    q_out = mp.Queue()
    proc = [mp.Process(target=fun, args=(f, q_in, q_out, args, kwargs))
            for _ in range(nprocs)]

    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]
