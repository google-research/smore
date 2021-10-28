# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    db_name = sys.argv[1]

    perfs = {}
    n_config = 0
    t_list = [500, 5000, 50000]
    methods = ['naive', 'sqrt']
    for method in methods:
        for t in t_list:
            l = []
            fname = os.path.join(db_name, '%s-%d.txt' % (method, t))
            if not os.path.isfile(fname):
                continue
            with open(fname, 'r') as f:
                for row in f:
                    tt = float(row.strip().split()[-1])
                    l.append(tt)
            perfs[method + str(t)] = l
            n_config = len(l)
    
    for i in range(n_config):
        for method in methods:
            y = []
            for t in t_list:
                if not method + str(t) in perfs:
                    break
                y.append(perfs[method + str(t)][i])
            plt.plot(t_list[:len(y)], y)
        plt.legend(methods)
        plt.savefig('%s/t-%d.pdf' % (db_name, i))
        plt.close()