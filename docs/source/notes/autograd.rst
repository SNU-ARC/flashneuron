.. _autograd-mechanics:

Autograd mechanics
==================

This note will present an overview of how autograd works and records the
operations. It's not strictly necessary to understand all this, but we recommend
getting familiar with it, as it will help you write more efficient, cleaner
programs, and can aid you in debugging.

.. _excluding-subgraphs:

Excluding subgraphs from backward
---------------------------------

Every Tensor has a flag: :attr:`requires_grad` that allows for fine grained
exclusion of subgraphs from gradient computation and can increase efficiency.

.. _excluding-requires_grad:

``requires_grad``
^^^^^^^^^^^^^^^^^

If there's a single input to an operation that requires gradient, its output
will also require gradient. Conversely, only if all inputs don't require
gradient, the output also won't require it. Backward computation is never
performed in the subgraphs, where all Tensors didn't require gradients.

.. code::

    >>> x = torch.randn(5, 5)  # requires_grad=False by default
    >>> y = torch.randn(5, 5)  # requires_grad=False by default
    >>> z = torch.randn((5, 5), requires_grad=True)
    >>> a = x + y
    >>> a.requires_grad
    False
    >>> b = a + z
    >>> b.requires_grad
    True

This is especially useful when you want to freeze part of your model, or you
know in advance that you're not going to use gradients w.r.t. some parameters.
For example if you want to finetune a pretrained CNN, it's enough to switch the
:attr:`requires_grad` flags in the frozen base, and no intermediate buffers will
be saved, until the computation gets to the last layer, where the affine
transform will use weights that require gradient, and the output of the network
will also require them.

.. code::

    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(512, 100)

    # Optimize only the classifier
    optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

.. _how-autograd-encodes-history:

How autograd encodes the history
--------------------------------

Autograd is reverse automatic differentiation system.  Conceptually,
autograd records a graph recording all of the operations that created
the data as you execute operations, giving you a directed acyclic graph
whose leaves are the input tensors and roots are the output tensors.
By tracing this graph from roots to leaves, you can automatically
compute the gradients using the chain rule.

Internally, autograd represents this graph as a graph of
:class:`Function` objects (really expressions), which can be
:meth:`~torch.autograd.Function.apply` ed to compute the result of
evaluating the graph.  When computing the forwards pass, autograd
simultaneously performs the requested computations and builds up a graph
representing the function that computes the gradient (the ``.grad_fn``
attribute of each :class:`torch.Tensor` is an entry point into this graph).
When the forwards pass is completed, we evaluate this graph in the
backwards pass to compute the gradients.

An important thing to note is that the graph is recreated from scratch at every
iteration, and this is exactly what allows for using arbitrary Python control
flow statements, that can change the overall shape and size of the graph at
every iteration. You don't have to encode all possible paths before you
launch the training - what you run is what you differentiate.

In-place operations with autograd
---------------------------------

Supporting in-place operations in autograd is a hard matter, and we discourage
their use in most cases. Autograd's aggressive buffer freeing and reuse makes
it very efficient and there are very few occasions when in-place operations
actually lower memory usage by any significant amount. Unless you're operating
under heavy memory pressure, you might never need to use them.

There are two main reasons that limit the applicability of in-place operations:

1. In-place operations can potentially overwrite values required to compute
   gradients.

2. Every in-place operation actually requires the implementation to rewrite the
   computational graph. Out-of-place versions simply allocate new objects and
   keep references to the old graph, while in-place operations, require
   changing the creator of all inputs to the :class:`Function` representing
   this operation. This can be tricky, especially if there are many Tensors
   that reference the same storage (e.g. created by indexing or transposing),
   and in-place functions will actually raise an error if the storage of
   modified inputs is referenced by any other :class:`Tensor`.

In-place correctness checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every tensor keeps a version counter, that is incremented every time it is
marked dirty in any operation. When a Function saves any tensors for backward,
a version counter of their containing Tensor is saved as well. Once you access
``self.saved_tensors`` it is checked, and if it is greater than the saved value
an error is raised. This ensures that if you're using in-place
functions and not seeing any errors, you can be sure that the computed
gradients are correct.

Multithreaded Autograd
----------------------

The autograd engine is responsible for running all the backward operations
necessary to compute the backward pass. This section will describe all the details
that can help you make the best use of it in a multithreaded environment.(this is
relevant only for PyTorch 1.6+ as the behavior in previous version was different).

User could train their model with multithreading code (e.g. Hogwild training), and
does not block on the concurrent backward computations, example code could be:

.. code::

    # Define a train function to be used in different threads
    def train_fn():
        x = torch.ones(5, 5, requires_grad=True)
        # forward
        y = (x + 3) * (x + 4) * 0.5
        # backward
        y.sum().backward()
        # potential optimizer update


    # User write their own threading code to drive the train_fn
    threads = []
    for _ in range(10):
        p = threading.Thread(target=train_fn, args=())
        p.start()
        threads.append(p)

    for p in threads:
        p.join()


Note that some behaviors that user should be aware of:

Concurrency on CPU
^^^^^^^^^^^^^^^^^^

When you run ``backward()`` or ``grad()`` via python or C++ API in multiple
threads on CPU, you are expecting to see extra concurrency instead of
serializing all the backward calls in a specific order during execution
(behavior before PyTorch 1.6).

Non-determinism
^^^^^^^^^^^^^^^

If you are calling ``backward()`` on multiple thread concurrently but with
shared inputs (i.e. Hogwild CPU training). Since parameters are automatically
shared across threads, gradient accumulation might become non-deterministic on
backward calls across threads, because two backward calls might access and try
to accumulate the same ``.grad`` attribute. This is technically not safe, and
it might result in racing condition and the result might be invalid to use.

But this is expected pattern if you are using the multithreading approach to
drive the whole training process but using shared parameters, user who use
multithreading should have the threading model in mind and should expect this
to happen. User could use the functional API :func:`torch.autograd.grad` to
calculate the gradients instead of ``backward()`` to avoid non-determinism.

Graph retaining
^^^^^^^^^^^^^^^

If part of the autograd graph is shared between threads, i.e. run first
part of forward single thread, then run second part in multiple threads,
then the first part of graph is shared. In this case different threads
execute ``grad()`` or ``backward()`` on the same graph might have issue of
destroying the graph on the fly of one thread, and the other thread will
crash in this case. Autograd will error out to the user similar to what call
``backward()`` twice with out ``retain_graph=True``, and let the user know
they should use ``retain_graph=True``.

Thread Safety on Autograd Node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since Autograd allows the caller thread to drive its backward execution for
potential parallelism, it's important that we ensure thread safety on CPU with
parallel backwards that share part/whole of the GraphTask.

Custom Python ``autograd.function`` is automatically thread safe because of GIL.
for built-in C++ Autograd Nodes(e.g. AccumulateGrad, CopySlices) and custom
``autograd::Function``, the Autograd Engine uses thread mutex locking to protect
thread safety on autograd Nodes that might have state write/read.

No thread safety on C++ hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Autograd relies on the user to write thread safe C++ hooks. If you want the hook
to be correctly applied in multithreading environment, you will need to write
proper thread locking code to ensure the hooks are thread safe.

.. _complex_autograd-doc:

Autograd for Complex Numbers
----------------------------

The short version:

- When you use PyTorch to differentiate any function :math:`f(z)` with complex domain and/or codomain,
  the gradients are computed under the assumption that the function is a part of a larger real-valued
  loss function :math:`g(input)=L`. The gradient computed is :math:`\frac{\partial L}{\partial z^*}`
  (note the conjugation of z), the negative of which is precisely the direction of steepest descent
  used in Gradient Descent algorithm. Thus, all the existing optimizers work out of
  the box with complex parameters.
- This convention matches TensorFlow's convention for complex
  differentiation, but is different from JAX (which computes
  :math:`\frac{\partial L}{\partial z}`).
- If you have a real-to-real function which internally uses complex
  operations, the convention here doesn't matter: you will always get
  the same result that you would have gotten if it had been implemented
  with only real operations.

If you are curious about the mathematical details, or want to know how
to define complex derivatives in PyTorch, read on.

What are complex derivatives?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mathematical definition of complex-differentiability takes the
limit definition of a derivative and generalizes it to operate on
complex numbers. Consider a function :math:`f: ℂ → ℂ`,

    .. math::
        `f(z=x+yj) = u(x, y) + v(x, y)j`

where :math:`u` and :math:`v` are two variable real valued functions.

Using the derivative definition, we can write:

    .. math::
        f'(z) = \lim_{h \to 0, h \in C} \frac{f(z+h) - f(z)}{h}

In order for this limit to exist, not only must :math:`u` and :math:`v` must be
real differentiable, but :math:`f` must also satisfy the Cauchy-Riemann `equations
<https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations>`_.  In
other words: the limit computed for real and imaginary steps (:math:`h`)
must be equal. This is a more restrictive condition.

The complex differentiable functions are commonly known as holomorphic
functions. They are well behaved, have all the nice properties that
you've seen from real differentiable functions, but are practically of no
use in the optimization world. For optimization problems, only real valued objective
functions are used in the research community since complex numbers are not part of any
ordered field and so having complex valued loss does not make much sense.

It also turns out that no interesting real-valued objective fulfill the
Cauchy-Riemann equations. So the theory with homomorphic function cannot be
used for optimization and most people therefore use the Wirtinger calculus.

Wirtinger Calculus comes in picture ...
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So, we have this great theory of complex differentiability and
holomorphic functions, and we can’t use any of it at all, because many
of the commonly used functions are not holomorphic. What’s a poor
mathematician to do? Well, Wirtinger observed that even if :math:`f(z)`
isn’t holomorphic, one could rewrite it as a two variable function
:math:`f(z, z*)` which is always holomorphic. This is because real and
imaginary of the components of :math:`z` can be expressed in terms of
:math:`z` and :math:`z^*` as:

    .. math::
        \begin{aligned}
            Re(z) &= \frac {z + z^*}{2} \\
            Im(z) &= \frac {z - z^*}{2j}
        \end{aligned}

Wirtinger calculus suggests to study :math:`f(z, z^*)` instead, which is
guaranteed to be holomorphic if :math:`f` was real differentiable (another
way to think of it is as a change of coordinate system, from :math:`f(x, y)`
to :math:`f(z, z^*)`.)  This function has partial derivatives
:math:`\frac{\partial }{\partial z}` and :math:`\frac{\partial}{\partial z^{*}}`.
We can use the chain rule to establish a
relationship between these partial derivatives and the partial
derivatives w.r.t., the real and imaginary components of :math:`z`.

    .. math::
        \begin{aligned}
            \frac{\partial }{\partial x} &= \frac{\partial z}{\partial x} * \frac{\partial }{\partial z} + \frac{\partial z^*}{\partial x} * \frac{\partial }{\partial z^*} \\
                                         &= \frac{\partial }{\partial z} + \frac{\partial }{\partial z^*}   \\
            \\
            \frac{\partial }{\partial y} &= \frac{\partial z}{\partial y} * \frac{\partial }{\partial z} + \frac{\partial z^*}{\partial y} * \frac{\partial }{\partial z^*} \\
                                         &= 1j * (\frac{\partial }{\partial z} - \frac{\partial }{\partial z^*})
        \end{aligned}

From the above equations, we get:

    .. math::
        \begin{aligned}
            \frac{\partial }{\partial z} &= 1/2 * (\frac{\partial }{\partial x} - 1j * \frac{\partial }{\partial y})   \\
            \frac{\partial }{\partial z^*} &= 1/2 * (\frac{\partial }{\partial x} + 1j * \frac{\partial }{\partial y})
        \end{aligned}

which is the classic definition of Wirtinger calculus that you would find on `Wikipedia <https://en.wikipedia.org/wiki/Wirtinger_derivatives>`_.

There are a lot of beautiful consequences of this change.

- For one, the Cauchy-Riemann equations translate into simply saying that :math:`\frac{\partial f}{\partial z^*} = 0` (that is to say, the function :math:`f` can be written
  entirely in terms of :math:`z`, without making reference to :math:`z^*`).
- Another important (and somewhat counterintuitive) result, as we’ll see later, is that when we do optimization on a real-valued loss, the step we should
  take while making variable update is given by :math:`\frac{\partial Loss}{\partial z^*}` (not :math:`\frac{\partial Loss}{\partial z}`).

For more reading, check out: https://arxiv.org/pdf/0906.4835.pdf

How is Wirtinger Calculus useful in optimization?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Researchers in audio and other fields, more commonly, use gradient
descent to optimize real valued loss functions with complex variables.
Typically, these people treat the real and imaginary values as separate
channels that can be updated. For a step size :math:`s/2` and loss
:math:`L`, we can write the following equations in :math:`ℝ^2`:

    .. math::
        \begin{aligned}
            x_{n+1} &= x_n - (s/2) * \frac{\partial L}{\partial x}  \\
            y_{n+1} &= y_n - (s/2) * \frac{\partial L}{\partial y}
        \end{aligned}

How do these equations translate into complex space :math:`ℂ`?

    .. math::
        \begin{aligned}
            z_{n+1} &= x_n - (s/2) * \frac{\partial L}{\partial x} + 1j * (y_n - (s/2) * \frac{\partial L}{\partial y}) \\
                    &= z_n - s * 1/2 * (\frac{\partial L}{\partial x} + j \frac{\partial L}{\partial y}) \\
                    &= z_n - s * \frac{\partial L}{\partial z^*}
        \end{aligned}

Something very interesting has happened: Wirtinger calculus tells us
that we can simplify the complex variable update formula above to only
refer to the conjugate Wirtinger derivative
:math:`\frac{\partial L}{\partial z^*}`, giving us exactly the step we take in optimization.

Because the conjugate Wirtinger derivative gives us exactly the correct step for a real valued loss function, PyTorch gives you this derivative
when you differentiate a function with a real valued loss.

How does PyTorch compute the conjugate Wirtinger derivative?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically, our derivative formulas take in `grad_output` as an input,
representing the incoming Vector-Jacobian product that we’ve already
computed, aka, :math:`\frac{\partial L}{\partial s^*}`, where :math:`L`
is the loss of the entire computation (producing a real loss) and
:math:`s` is the output of our function. The goal here is to compute
:math:`\frac{\partial L}{\partial z^*}`, where :math:`z` is the input of
the function.  It turns out that in the case of real loss, we can
get away with *only* calculating :math:`\frac{\partial L}{\partial z^*}`,
even though the chain rule implies that we also need to
have access to :math:`\frac{\partial L}{\partial z^*}`.  If you want
to skip this derivation, look at the last equation in this section
and then skip to the next section.

Let’s continue working with :math:`f: ℂ → ℂ` defined as
:math:`f(z) = f(x+yj) = u(x, y) + v(x, y)j`. As discussed above,
autograd’s gradient convention is centered around optimization for real
valued loss functions, so let’s assume :math:`f` is a part of larger
real valued loss function :math:`g`. Using chain rule, we can write:

    .. math::
        \frac{\partial L}{\partial z^*} = \frac{\partial L}{\partial u} * \frac{\partial u}{\partial z^*} + \frac{\partial L}{\partial v} * \frac{\partial v}{\partial z^*}
        :label: [1]

Now using Wirtinger derivative definition, we can write:

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial s} = 1/2 * (\frac{\partial L}{\partial u} - \frac{\partial L}{\partial v} j) \\
            \frac{\partial L}{\partial s^*} = 1/2 * (\frac{\partial L}{\partial u} + \frac{\partial L}{\partial v} j)
        \end{aligned}

It should be noted here that since :math:`u` and :math:`v` are real
functions, and :math:`L` is real by our assumption that :math:`f` is a
part of a real valued function, we have:

    .. math::
        (\frac{\partial L}{\partial s})^* = \frac{\partial L}{\partial s^*}
        :label: [2]

i.e., :math:`\frac{\partial L}{\partial s}` equals to :math:`grad\_output^*`.

Solving the above equations for :math:`\frac{\partial L}{\partial u}` and :math:`\frac{\partial L}{\partial v}`, we get:

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial u} = \frac{\partial L}{\partial s} + \frac{\partial L}{\partial s^*} \\
            \frac{\partial L}{\partial v} = -1j * (\frac{\partial L}{\partial s} - \frac{\partial L}{\partial s^*})
        \end{aligned}
        :label: [3]

Substituting :eq:`[3]` in :eq:`[1]`, we get:

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial z^*} &= (\frac{\partial L}{\partial s} + \frac{\partial L}{\partial s^*}) * \frac{\partial u}{\partial z^*} - 1j * (\frac{\partial L}{\partial s} - \frac{\partial L}{\partial s^*}) * \frac{\partial v}{\partial z^*}  \\
                                            &= \frac{\partial L}{\partial s} * (\frac{\partial u}{\partial z^*} + \frac{\partial v}{\partial z^*} j) + \frac{\partial L}{\partial s^*} * (\frac{\partial u}{\partial z^*} - \frac{\partial v}{\partial z^*} j)  \\
                                            &= \frac{\partial L}{\partial s^*} * \frac{\partial (u + vj)}{\partial z^*} + \frac{\partial L}{\partial s} * \frac{\partial (u + vj)^*}{\partial z^*}  \\
                                            &= \frac{\partial L}{\partial s} * \frac{\partial s}{\partial z^*} + \frac{\partial L}{\partial s^*} * \frac{\partial s^*}{\partial z^*}    \\
        \end{aligned}

Using :eq:`[2]`, we get:

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial z^*} &= (\frac{\partial L}{\partial s^*})^* * \frac{\partial s}{\partial z^*} + \frac{\partial L}{\partial s^*} * (\frac{\partial s}{\partial z})^*  \\
                                            &= \boxed{ (grad\_output)^* * \frac{\partial s}{\partial z^*} + grad\_output * {(\frac{\partial s}{\partial z})}^* }       \\
        \end{aligned}
        :label: [4]

This last equation is the important one for writing your own gradients,
as it decomposes our derivative formula into a simpler one that is easy
to compute by hand.

How can I write my own derivative formula for a complex function?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The above boxed equation gives us the general formula for all
derivatives on complex functions.  However, we still need to
compute :math:`\frac{\partial s}{\partial z}` and :math:`\frac{\partial s}{\partial z^*}`.
There are two ways you could do this:

    - The first way is to just use the definition of Wirtinger derivatives directly and calculate :math:`\frac{\partial s}{\partial z}` and :math:`\frac{\partial s}{\partial z^*}` by
      using :math:`\frac{\partial s}{\partial x}` and :math:`\frac{\partial s}{\partial y}`
      (which you can compute in the normal way).
    - The second way is to use the change of variables trick and rewrite :math:`f(z)` as a two variable function :math:`f(z, z^*)`, and compute
      the conjugate Wirtinger derivatives by treating :math:`z` and :math:`z^*` as independent variables. This is often easier; for example, if the function in question is holomorphic, only :math:`z` will be used (and :math:`\frac{\partial s}{\partial z^*}` will be zero).

Let's consider the function :math:`f(z = x + yj) = c * z = c * (x+yj)` as an example, where :math:`c \in ℝ`.

Using the first way to compute the Wirtinger derivatives, we have.

.. math::
    \begin{aligned}
        \frac{\partial s}{\partial z} &= 1/2 * (\frac{\partial s}{\partial x} - \frac{\partial s}{\partial y} j) \\
                                      &= 1/2 * (c - (c * 1j) * 1j)  \\
                                      &= c                          \\
        \\
        \\
        \frac{\partial s}{\partial z^*} &= 1/2 * (\frac{\partial s}{\partial x} + \frac{\partial s}{\partial y} j) \\
                                        &= 1/2 * (c + (c * 1j) * 1j)  \\
                                        &= 0                          \\
    \end{aligned}

Using :eq:`[4]`, and `grad\_output = 1.0` (which is the default grad output value used when :func:`backward` is called on a scalar output in PyTorch), we get:

    .. math::
        \frac{\partial L}{\partial z^*} = 1 * 0 + 1 * c = c

Using the second way to compute Wirtinger derivatives, we directly get:

    .. math::
        \begin{aligned}
           \frac{\partial s}{\partial z} &= \frac{\partial (c*z)}{\partial z}       \\
                                         &= c                                       \\
            \frac{\partial s}{\partial z^*} &= \frac{\partial (c*z)}{\partial z^*}       \\
                                         &= 0
        \end{aligned}

And using :eq:`[4]` again, we get :math:`\frac{\partial L}{\partial z^*} = c`. As you can see, the second way involves lesser calculations, and comes
in more handy for faster calculations.

What about cross-domain functions?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some functions map from complex inputs to real outputs, or vice versa.
These functions form a special case of :eq:`[4]`, which we can derive using the
chain rule:

    - For :math:`f: ℂ → ℝ`, we get:

        .. math::
            \frac{\partial L}{\partial z^*} = 2 * grad\_output * \frac{\partial s}{\partial z^{*}}

    - For :math:`f: ℝ → ℂ`, we get:

        .. math::
            \frac{\partial L}{\partial z^*} = 2 * Re(grad\_out^* * \frac{\partial s}{\partial z^{*}})
