#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 17, 2023
# Note that we referred to https://github.com/JaxGaussianProcesses/GPJax/blob/main/tests/integration_tests.py for implementing this test.
#

import numpy as np
import jupytext


class Result:
    def __init__(self, path, comparisons):
        self.path = path
        self.comparisons = comparisons

    def _compare(self, observed_variables, variable_name, true_value, operation):
        try:
            value = operation(observed_variables[variable_name])
            assert true_value == value
        except AssertionError as e:
            print(self.path)
            print(e)
            print(true_value, value)

    def test(self):
        notebook = jupytext.read(self.path)
        contents = ""

        for c in notebook["cells"]:
            if c["cell_type"] == "code":
                contents += c["source"]
                contents += '\n\n'

        lines = contents.split('\n')
        new_lines = []

        for line in lines:
            if 'utils_plotting' not in line:
                new_lines.append(line)
        lines = new_lines

        contents = "\n".join([line for line in lines])

        loc = {}
        exec(contents, globals(), loc)
        for k, v in self.comparisons:
            truth, op = v
            self._compare(
                observed_variables=loc, variable_name=k, true_value=truth, operation=op
            )


result = Result(
    path='../examples/01_basics/example_basics_bo.py',
    comparisons=[
        ['Y_train', (-1.2749386303305092, np.min)],
        ['Y_train', (-1.274187339080934, lambda Y: Y[-1, 0])],
    ],
)
result.test()

result = Result(
    path='../examples/01_basics/example_basics_gp.py',
    comparisons=[
        ['mu', (-0.04591079227334406, lambda Y: Y[-1, 0])],
        ['mu', (13.937631558319126, np.sum)],
        ['sigma', (0.4951346343515692, lambda Y: Y[-1, 0])],
        ['sigma', (68.65751708043783, np.sum)],
        ['Sigma', (690.8686182390853, np.sum)],
    ],
)
result.test()

result = Result(
    path='../examples/03_bo/example_bo_aei.py',
    comparisons=[
        ['Y_train', (-1.2749784051487447, np.min)],
        ['Y_train', (-1.2749784051487447, lambda Y: Y[-1, 0])],
    ],
)
result.test()

result = Result(
    path='../examples/03_bo/example_bo_ei.py',
    comparisons=[
        ['Y_train', (-1.2749940277251608, np.min)],
        ['Y_train', (-1.27389177011487, lambda Y: Y[-1, 0])],
    ],
)
result.test()

result = Result(
    path='../examples/03_bo/example_bo_pi.py',
    comparisons=[
        ['Y_train', (-0.7487013181048252, np.min)],
        ['Y_train', (-0.7487013181048252, lambda Y: Y[-1, 0])],
    ],
)
result.test()

result = Result(
    path='../examples/03_bo/example_bo_pure_exploit.py',
    comparisons=[
        ['Y_train', (-0.8991587924341444, np.min)],
        ['Y_train', (-0.8991587924341444, lambda Y: Y[-1, 0])],
    ],
)
result.test()

result = Result(
    path='../examples/03_bo/example_bo_pure_explore.py',
    comparisons=[
        ['Y_train', (-1.2089305786421192, np.min)],
        ['Y_train', (18.506807094195395, lambda Y: Y[-1, 0])],
    ],
)
result.test()

result = Result(
    path='../examples/03_bo/example_bo_ucb.py',
    comparisons=[
        ['Y_train', (-1.2746395000914887, np.min)],
        ['Y_train', (-1.2741712887239314, lambda Y: Y[-1, 0])],
    ],
)
result.test()
