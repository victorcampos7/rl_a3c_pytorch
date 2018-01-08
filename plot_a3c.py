from __future__ import absolute_import
from __future__ import print_function

import re
import pandas
import argparse
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')


def _parse_line(line):
    line = line.split(' ', 3)[-1]
    
    # Parse timestamp
    time_, ep_reward_, ep_length_, mean_reward_ = [l.strip() for l in line.split(',')]
    match = re.compile(r"Time (?P<h>.*?)h (?P<m>.*?)m (?P<s>.*?)s", re.VERBOSE).match(time_)
    t = float(match.group('h')) + float(match.group('m'))/60. + float(match.group('s'))/3600.

    # Parse remaining info
    ep_reward = float(ep_reward_.split()[-1])
    ep_length = float(ep_length_.split()[-1])
    mean_reward = float(mean_reward_.split()[-1])

    return t, ep_reward, ep_length, mean_reward


def _line_needs_parsing(line):
    return re.search(r'\bTime\b', line) and re.search(r'\bepisode\b', line)


def _parse_file(file_path, smoothing_factor):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    parsed_data = [_parse_line(line) for line in lines if _line_needs_parsing(line)]
    parsed_data = [list(t) for t in zip(*parsed_data)]
    t, parsed_data = parsed_data[0], parsed_data[1:]
    
    # The code resets the time after 24h
    prev_t = -1
    offset = 0
    for idx, t_ in enumerate(t):
    	if t_ < prev_t:
    		offset += 24
    	prev_t = t_
    	t[idx] += offset

    parsed_data = [_exponential_moving_average_smoothing(d, smoothing_factor) for d in parsed_data]
    ep_reward, ep_length, mean_reward = parsed_data
    return t, ep_reward, ep_length, mean_reward


def _generate_plots(t, ep_reward, ep_length, mean_reward, labels, output_file, title):
    # Create subplots
    fig, axarr = plt.subplots(1, sharex=True, figsize=(12, 9))

    colormap = plt.cm.Set2
    axarr.set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(t))])

    for t_, ep_reward_, ep_length_, mean_reward_, label in zip(t, ep_reward, ep_length, mean_reward, labels):
        axarr.plot(t_, ep_reward_, label=label)
        axarr.set_xlabel('Training time (hours)')
        axarr.set_ylabel('Reward')
        axarr.grid('on')

    axarr.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), fancybox=True, shadow=True, ncol=1)

    if title:
    	axarr.set_title(title)

    if output_file is not None:
        fig.savefig(output_file, bbox_inches='tight')
    else:
        plt.show()


def _exponential_moving_average_smoothing(values, smoothing_factor):
    if smoothing_factor == 0:
        return values
    if isinstance(values, list):
        values = np.array(values)
    return pandas.stats.moments.ewma(values, span=smoothing_factor)


def _parse_arguments():
    parser = argparse.ArgumentParser('Plot training log for the addition task')
    parser.add_argument('--logs', nargs='+', required=True, help='List of files to parse and plot')
    parser.add_argument('--labels', nargs='+', required=True, help='List of labels for the plot legend')
    parser.add_argument('--smoothing_factor', default=0, type=int, help='Exponential moving average smoothing factor')
    parser.add_argument('--output_file', default=None, help='(Optional) Path to store the figure')
    parser.add_argument('--title', default=None, help='(Optional) Plot title (e.g. environment name)')
    return parser.parse_args()


def main():
    args = _parse_arguments()
    values = [list() for _ in range(4)]

    for log in args.logs:
        parsed_values = _parse_file(log, args.smoothing_factor)
        for v, p in zip(values, parsed_values):
            v.append(p)

    _generate_plots(t=values[0],
                    ep_reward=values[1],
                    ep_length=values[2],
                    mean_reward=values[3],
                    labels=args.labels,
                    output_file=args.output_file,
                    title=args.title)


if __name__ == '__main__':
    main()
