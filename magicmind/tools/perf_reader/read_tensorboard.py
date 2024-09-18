import os
import argparse
import kernel_stats_pb2
import prettytable
from textwrap3 import fill


class ProtoParse(object):
    """Object for pase/print perf proto"""

    def __init__(self):
        self.kernel_util = []
        self.kernel_num = 0
        self.sum_duration = 0

    def to_dict(self, kernel_report):
        """To gen dict for kernels"""
        kernel_info = {}
        self.kernel_num += 1
        kernel_info["rank"] = self.kernel_num
        kernel_info["kernel_name"] = kernel_report.name
        kernel_info["layer_name"] = kernel_report.op_name
        # compute average duration and convert ns to us
        avg_duration_us = round(
            kernel_report.total_duration_ns / 1000. / kernel_report.occurrences,
            2)
        kernel_info["avg_duration_us"] = avg_duration_us
        self.sum_duration += avg_duration_us
        self.kernel_util.append(kernel_info)

    def parse(self, message_obj):
        """To parse pb file"""
        self.message_obj = message_obj
        for kernel_report in self.message_obj.reports:
            # convert proto fmt to dict
            self.to_dict(kernel_report)

    def print_info(self, title, topk):
        """To print kernel durations"""
        kernel_table = prettytable.PrettyTable(hrules=prettytable.ALL)
        kernel_table.title = title
        kernel_table.field_names = [
            "rank", "kernel_name", "avg_duration(us)", "percentage",
            "layer_name"
        ]
        if topk < len(self.kernel_util):
            self.kernel_util = self.kernel_util[:topk]
        for kernel_info in self.kernel_util:
            kernel_percentage = round(
                (kernel_info["avg_duration_us"] * 100. / self.sum_duration), 2)
            kernel_table.add_row([
                kernel_info["rank"],
                fill(kernel_info["kernel_name"], width=50),
                kernel_info["avg_duration_us"],
                str(kernel_percentage) + "%",
                fill(kernel_info["layer_name"], width=80)
            ])
        print(kernel_table)


def print_kernel(path, top):
    """To read a kernel pb and generate top table"""
    raw_data = ""
    if not os.path.isdir(path) and path.endswith("kernel_stats.pb"):
        try:
            with open(path, 'rb') as f:
                raw_data = f.read()
        except Exception as e:
            print("Can't read kernel_stats file path: %s, Error %s", path, e)
    else:
        print("Invalid proto format.")

    kernel_db = kernel_stats_pb2.KernelStatsDb()
    kernel_db.ParseFromString(raw_data)
    proto_parse = ProtoParse()
    proto_parse.parse(kernel_db)
    proto_parse.print_info(path, top)


def search(path):
    """To search all kernel pb files belongs to a path"""
    path_list = []
    for filename in os.listdir(path):
        fp = os.path.join(path, filename)
        if os.path.isfile(fp) and filename.endswith("kernel_stats.pb"):
            path_list.append(fp)
        elif os.path.isdir(fp):
            path_list.extend(search(fp))
    return path_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto_path", help="kernel stats proto path")
    parser.add_argument("--perf_path", help="profiler proto path")
    parser.add_argument(
        "--top",
        default=10,
        help="number of most time-consuming kernels, default: 10")
    args = parser.parse_args()
    if args.proto_path:
        print_kernel(args.proto_path, int(args.top))
    elif args.perf_path:
        path_list = search(args.perf_path)
        if not path_list:
            print("Can't find kernel_stats file path: %s", args.perf_path)
        for p in path_list:
            print_kernel(p, int(args.top))
    else:
        print("Must provide proto path or perf path!")
