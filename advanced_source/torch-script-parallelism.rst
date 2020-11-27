TorchScript의 동적 병렬 처리(DYNAMIC PARALLELISM)
==================================================

이 튜토리얼에서는, 우리는 TorchScript에서 *dynamic inter-op parallelism* 를 하는 구문(syntax)을 소개합니다.
이 병렬처리에는 다음과 같은 속성이 있습니다:

* 동적(dynamic) - 생성된 병렬 작업의 수와 작업 부하는 프로그램의 제어 흐름에 따라 달라질 수 있습니다.
* inter-op - 병렬 처리는 TorchScript 프로그램 조각을 병렬로 실행하는 것과 관련이 있습니다. 이는 개별 연산자를 분할하고 연산자 작업의 하위 집합을 병렬로 실행하는 것과
관계되는 *intra-op parallelism* 와는 구별됩니다.

기본 구문
------------

dynamic 병렬 처리를 위한 두가지 중요한 API는 다음과 같습니다:

* ``torch.jit.fork(fn : Callable[..., T], *args, **kwargs) -> torch.jit.Future[T]``
* ``torch.jit.wait(fut : torch.jit.Future[T]) -> T``

예제를 통해 이러한 작동 방식을 보여주는 좋은 방법:

.. code-block:: python

    import torch

    def foo(x):
        return torch.neg(x)

    @torch.jit.script
    def example(x):
        # 병렬 처리를 사용하여 `foo`를 호출:
        # 먼저, 작업을 "fork" 합니다. 이 작업은 `x` 인자(argument)와 함께 `foo` 를 실행합니다
        future = torch.jit.fork(foo, x)

        # 일반적으로 `foo` 호출
        x_normal = foo(x)

        # 둘째, 작업이 실행 중일 수 있으므로 우리는 작업을 "기다립니다".
        # 병렬로, 결과를 사용할 수 있을 때까지 "대기" 해야합니다.
        # "fork()" 와 "wait()" 사이에 코드 라인이 있음에 유의하십시오.
        # 주어진 Future를 호출하면, 계산을 오버랩(overlap)해서 병렬로 실행할 수 있습니다.
        x_parallel = torch.jit.wait(future)

        return x_normal, x_parallel

    print(example(torch.ones(1))) # (-1., -1.)


``fork()`` 는 호출 가능한 ``fn`` 과 해당 호출 가능한  ``args`` 및  ``kwargs`` 에 대한 인자를 취하고  ``fn`` 실행을 위한 비동기(asynchronous) 작업을 생성합니다.
``fn`` 은 함수, 메소드, 또는 모듈 인스턴스일 수 있습니다. ``fork()`` 는  ``Future`` 라고 불리는 이 실행 결과의 값에 대한 참조(reference)를 반환합니다.
``fork`` 는 비동기 작업을 생성한 직후에 반환되기 때문에,  ``fork()`` 호출 후 코드 라인이 실행될 때까지 ``fn`` 이 실행되지 않을 수 있습니다.
따라서, ``wait()`` 은 비동기 작업이 완료 될때까지 대기하고 값을 반환하는데 사용됩니다.

이러한 구조는 함수 내에서 명령문 실행을 오버랩하거나 (작업된 예제 섹션에 표시됨) 루프와 같은 다른 언어 구조로 구성 될 수 있습니다:

.. code-block:: python

    import torch
    from typing import List

    def foo(x):
        return torch.neg(x)

    @torch.jit.script
    def example(x):
        futures : List[torch.jit.Future[torch.Tensor]] = []
        for _ in range(100):
            futures.append(torch.jit.fork(foo, x))

        results = []
        for future in futures:
            results.append(torch.jit.wait(future))

        return torch.sum(torch.stack(results))

    print(example(torch.ones([])))

.. note::

    Future의 빈 리스트(list)를 초기화할때, 우리는 명시적 유형 주석을  ``futures`` 에 추가해야 했습니다.
    TorchScript에서 빈 컨테이너(container)는 기본적으로 tensor 값을 포함한다고 가정하므로
    리스트 생성자(constructor) #에  ``List[torch.jit.Future[torch.Tensor]]`` 유형의 주석을 달았습니다.

이 예제는  ``fork()`` 를 사용하여 함수  ``foo`` 의 인스턴스 100개를 시작하고, 100개의 작업이 완료 될때까지
대기한 다음, 결과를 합산하여  ``-100.0`` 을 반환합니다.

Applied Example: Ensemble of Bidirectional LSTMs
------------------------------------------------

Let's try to apply parallelism to a more realistic example and see what sort
of performance we can get out of it. First, let's define the baseline model: an
ensemble of bidirectional LSTM layers.

.. code-block:: python

    import torch, time

    # In RNN parlance, the dimensions we care about are:
    # # of time-steps (T)
    # Batch size (B)
    # Hidden size/number of "channels" (C)
    T, B, C = 50, 50, 1024

    # A module that defines a single "bidirectional LSTM". This is simply two
    # LSTMs applied to the same sequence, but one in reverse
    class BidirectionalRecurrentLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell_f = torch.nn.LSTM(input_size=C, hidden_size=C)
            self.cell_b = torch.nn.LSTM(input_size=C, hidden_size=C)

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            # Forward layer
            output_f, _ = self.cell_f(x)

            # Backward layer. Flip input in the time dimension (dim 0), apply the
            # layer, then flip the outputs in the time dimension
            x_rev = torch.flip(x, dims=[0])
            output_b, _ = self.cell_b(torch.flip(x, dims=[0]))
            output_b_rev = torch.flip(output_b, dims=[0])

            return torch.cat((output_f, output_b_rev), dim=2)


    # An "ensemble" of `BidirectionalRecurrentLSTM` modules. The modules in the
    # ensemble are run one-by-one on the same input then their results are
    # stacked and summed together, returning the combined result.
    class LSTMEnsemble(torch.nn.Module):
        def __init__(self, n_models):
            super().__init__()
            self.n_models = n_models
            self.models = torch.nn.ModuleList([
                BidirectionalRecurrentLSTM() for _ in range(self.n_models)])

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            results = []
            for model in self.models:
                results.append(model(x))
            return torch.stack(results).sum(dim=0)

    # For a head-to-head comparison to what we're going to do with fork/wait, let's
    # instantiate the model and compile it with TorchScript
    ens = torch.jit.script(LSTMEnsemble(n_models=4))

    # Normally you would pull this input out of an embedding table, but for the
    # purpose of this demo let's just use random data.
    x = torch.rand(T, B, C)

    # Let's run the model once to warm up things like the memory allocator
    ens(x)

    x = torch.rand(T, B, C)

    # Let's see how fast it runs!
    s = time.time()
    ens(x)
    print('Inference took', time.time() - s, ' seconds')

On my machine, this network runs in ``2.05`` seconds. We can do a lot better!

Parallelizing Forward and Backward Layers
-----------------------------------------

A very simple thing we can do is parallelize the forward and backward layers
within ``BidirectionalRecurrentLSTM``. For this, the structure of the computation
is static, so we don't actually even need any loops. Let's rewrite the ``forward``
method of ``BidirectionalRecurrentLSTM`` like so:

.. code-block:: python

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            # Forward layer - fork() so this can run in parallel to the backward
            # layer
            future_f = torch.jit.fork(self.cell_f, x)

            # Backward layer. Flip input in the time dimension (dim 0), apply the
            # layer, then flip the outputs in the time dimension
            x_rev = torch.flip(x, dims=[0])
            output_b, _ = self.cell_b(torch.flip(x, dims=[0]))
            output_b_rev = torch.flip(output_b, dims=[0])

            # Retrieve the output from the forward layer. Note this needs to happen
            # *after* the stuff we want to parallelize with
            output_f, _ = torch.jit.wait(future_f)

            return torch.cat((output_f, output_b_rev), dim=2)

In this example, ``forward()`` delegates execution of ``cell_f`` to another thread,
while it continues to execute ``cell_b``. This causes the execution of both the
cells to be overlapped with each other.

Running the script again with this simple modification yields a runtime of
``1.71`` seconds for an improvement of ``17%``!

Aside: Visualizing Parallelism
------------------------------

We're not done optimizing our model but it's worth introducing the tooling we
have for visualizing performance. One important tool is the `PyTorch profiler <https://pytorch.org/docs/stable/autograd.html#profiler>`_.

Let's use the profiler along with the Chrome trace export functionality to
visualize the performance of our parallelized model:

.. code-block:: python
    with torch.autograd.profiler.profile() as prof:
        ens(x)
    prof.export_chrome_trace('parallel.json')

This snippet of code will write out a file named ``parallel.json``. If you
navigate Google Chrome to ``chrome://tracing``, click the ``Load`` button, and
load in that JSON file, you should see a timeline like the following:

.. image:: https://i.imgur.com/rm5hdG9.png

The horizontal axis of the timeline represents time and the vertical axis
represents threads of execution. As we can see, we are running two ``lstm``
instances at a time. This is the result of our hard work parallelizing the
bidirectional layers!

Parallelizing Models in the Ensemble
------------------------------------

You may have noticed that there is a further parallelization opportunity in our
code: we can also run the models contained in ``LSTMEnsemble`` in parallel with
each other. The way to do that is simple enough, this is how we should change
the ``forward`` method of ``LSTMEnsemble``:

.. code-block:: python

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            # Launch tasks for each model
            futures : List[torch.jit.Future[torch.Tensor]] = []
            for model in self.models:
                futures.append(torch.jit.fork(model, x))

            # Collect the results from the launched tasks
            results : List[torch.Tensor] = []
            for future in futures:
                results.append(torch.jit.wait(future))

            return torch.stack(results).sum(dim=0)

Or, if you value brevity, we can use list comprehensions:

.. code-block:: python

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            futures = [torch.jit.fork(model, x) for model in self.models]
            results = [torch.jit.wait(fut) for fut in futures]
            return torch.stack(results).sum(dim=0)

Like described in the intro, we've used loops to fork off tasks for each of the
models in our ensemble. We've then used another loop to wait for all of the
tasks to be completed. This provides even more overlap of computation.

With this small update, the script runs in ``1.4`` seconds, for a total speedup
of ``32%``! Pretty good for two lines of code.

We can also use the Chrome tracer again to see where's going on:

.. image:: https://i.imgur.com/kA0gyQm.png

We can now see that all ``LSTM`` instances are being run fully in parallel.

Conclusion
----------

In this tutorial, we learned about ``fork()`` and ``wait()``, the basic APIs
for doing dynamic, inter-op parallelism in TorchScript. We saw a few typical
usage patterns for using these functions to parallelize the execution of
functions, methods, or ``Modules`` in TorchScript code. Finally, we worked through
an example of optimizing a model using this technique and explored the performance
measurement and visualization tooling available in PyTorch.
