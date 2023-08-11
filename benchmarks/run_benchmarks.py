import subprocess
import re
import sys

CONFIG_FILE = 'benchmarks_configurations.json'

time_regex = re.compile(r'Elapsed: (.*)\n')
unit_regex = re.compile(r'(.*\.[0-9][0-9])(ns|μs|ms|s)')

time_conversions = {
    'ns': 1e-6,
    'μs': 1e-3,
    'ms': 1,
    's': 1e3
}

benchmark_specific = {
    'convolution': ['-k 3'],
    'finite_impulse_response_filter': [
        '-k 16',
        '-k 32',
        '-k 64',
    ],
    'max_pooling': ['--stride 2'],
    'fast_fourier_transform_window': [
        '-w 8',
        '-w 128'
    ],
}

verifiable_benchmarks = ['relu', 'softmax', 'convolution', 'matrix_multiplication', 'max_pooling']

size = [1024, 2048, 4096]
iterations = 5


def main():
    """
    Here we run only for the benchmarking configurations same as the excel file
    So features 1d and features float for all but matrix_multiplication
    -s 1024, 2048 and 4096
    5 iterations for each
    we get the benchmark name as an argument
    in the end remove most time and least time and average the rest
    print to stdout size: avg time
    """

    # get the name of the benchmark from the command line
    benchmark_name = sys.argv[1]

    if benchmark_name == 'matrix_multiplication':
        print('matrix_multiplication is not supported')
        return
    
    if benchmark_name not in verifiable_benchmarks:
        print('benchmark results might be incorrect')

    for s in size:
        execution_times = []
        command = 'cargo run --release --features 1d --features float --bin ' + benchmark_name + ' -- -s ' + str(s) + ' -t -v'
        args = ['']
        if benchmark_name in benchmark_specific:
            args = benchmark_specific[benchmark_name]
        for arg in args:
            command += ' ' + arg
            print(command)
            for _ in range(iterations):
                res = subprocess.run(command, stdout=subprocess.PIPE, shell=True, stderr = subprocess.DEVNULL)
                output = res.stdout.decode('utf-8')
                if benchmark_name in verifiable_benchmarks and 'passed' not in output:
                    print('benchmark failed')
                    return
                time = time_regex.search(output).group(1)
                time, unit = unit_regex.search(time).groups()
                if unit not in time_conversions:
                    print('unit not supported')
                    return
                execution_times.append(float(time) * time_conversions[unit])

            
        execution_times.sort()
        print(execution_times)
        execution_times = execution_times[1:-1]
        print(str(s) + ': ' + str(sum(execution_times) / len(execution_times)))        

if __name__ == "__main__":
    main()



"""
with open(CONFIG_FILE) as config_file:
    data = json.load(config_file)

for benchmark in data:
    if benchmark['benchmark'] not in ['convolution']:
        continue
    features_combinations = list(itertools.product(*[values for (_, values) in benchmark['features'].items()]))
    params = []
    for (k, values) in benchmark['parameters'].items():
        params.append([(k, value) for value in values])
    parameters_combinations = list(itertools.product(*params))

    for features in features_combinations:
        for parameters in parameters_combinations:
            for _ in range(3):
                command = 'cargo run --release ' + ' '.join(['--features ' + feature for feature in features]) + ' --bin ' + benchmark['benchmark'] + ' -- ' + ' '.join([p[0] + ' ' + str(p[1]) for p in parameters]) + ' -t'
                print(command)
                res = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
                l = execution_times.get((features, parameters))
                if l == None:
                    execution_times[(features, parameters)] = [time_regex.search(res.stdout.decode('utf-8')).group(1)]
                else:
                    l.append(time_regex.search(res.stdout.decode('utf-8')).group(1))
            

print(execution_times)
"""