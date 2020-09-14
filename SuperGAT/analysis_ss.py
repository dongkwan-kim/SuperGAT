from collections import deque
from pprint import pprint

import numpy as np
from scipy import stats
import os

from termcolor import cprint

from arguments import get_args


def simulate_early_stop(val_loss_matrix, val_perf_matrix, test_perf_matrix,
                        patience,
                        queue_length,
                        early_stop_threshold_loss,
                        early_stop_threshold_perf,
                        hard_total_epochs):
    patience, queue_length = int(patience), int(queue_length)
    hard_total_epochs = int(hard_total_epochs)

    test_perf_at_best_val_list = []

    for val_loss_list, val_perf_list, test_perf_list in zip(val_loss_matrix, val_perf_matrix, test_perf_matrix):

        val_loss_deque = deque(maxlen=queue_length)
        val_perf_deque = deque(maxlen=queue_length)

        best_val_perf = 0.
        test_perf_at_best_val = 0.
        best_test_perf_at_best_val = 0.

        for epoch, v_loss, v_perf, t_perf in zip(range(hard_total_epochs),
                                                 val_loss_list,
                                                 val_perf_list,
                                                 test_perf_list):

            if v_perf >= best_val_perf:
                test_perf_at_best_val = t_perf
                if test_perf_at_best_val > best_test_perf_at_best_val:
                    best_test_perf_at_best_val = test_perf_at_best_val

            if epoch > 0 and epoch > patience:

                recent_val_loss_mean = float(np.mean(val_loss_deque))
                val_loss_change = abs(recent_val_loss_mean - v_loss) / recent_val_loss_mean
                if val_loss_change < early_stop_threshold_loss:
                    test_perf_at_best_val_list.append(test_perf_at_best_val)
                    break

                recent_val_perf_mean = float(np.mean(val_perf_deque))
                val_perf_change = abs(recent_val_perf_mean - v_perf) / recent_val_perf_mean
                if val_perf_change < early_stop_threshold_perf:
                    test_perf_at_best_val_list.append(test_perf_at_best_val)
                    break

            val_loss_deque.append(v_loss)
            val_perf_deque.append(v_perf)

        else:
            test_perf_at_best_val_list.append(test_perf_at_best_val)

    return test_perf_at_best_val_list


def load_populations(dataset_name,
                     base_path="../logs-2020-neurips/log_when_to_stop/_recorded",
                     filter_func=None):
    result_info_dict = {}
    data_path = os.path.join(base_path, dataset_name)
    for i, exp_dir in enumerate(os.listdir(data_path)):

        if not filter_func(exp_dir):
            continue

        full_path = os.path.join(data_path, exp_dir)
        exp_tuple = exp_dir.split("+")[0].split("-")[:3]
        try:
            args_kwargs = {
                "model_name": exp_tuple[0],
                "dataset_class": "Planetoid" if exp_tuple[1] != "PPI" else "PPI",
                "dataset_name": exp_tuple[1],
                "custom_key": exp_tuple[2] + ("" if exp_tuple[1] != "PubMed" else "-500") + "-ES",
            }
            args = get_args(**args_kwargs)

            result_info = {"full_path": full_path, "args": args}
            for perf_file in os.listdir(full_path):
                if "val_loss" in perf_file:
                    result_info["val_loss"] = os.path.join(full_path, perf_file)
                elif "val_perf" in perf_file:
                    result_info["val_perf"] = os.path.join(full_path, perf_file)
                elif "test_perf" in perf_file:
                    result_info["test_perf"] = os.path.join(full_path, perf_file)
            result_info_dict[exp_dir] = result_info
        except Exception as e:
            cprint(f"Exception in {full_path} and {exp_tuple}, {e}", "red")

    for exp_dir, v in result_info_dict.items():
        args = v["args"]
        val_loss_matrix = np.load(v["val_loss"], allow_pickle=True)
        val_perf_matrix = np.load(v["val_perf"], allow_pickle=True)
        test_perf_matrix = np.load(v["test_perf"], allow_pickle=True)

        if "PPI" in exp_dir:
            val_loss_matrix = val_loss_matrix[:30, :]
            val_perf_matrix = val_perf_matrix[:30, :]
            test_perf_matrix = test_perf_matrix[:30, :]

        test_perf_at_best_val_list = simulate_early_stop(val_loss_matrix, val_perf_matrix, test_perf_matrix,
                                                         args.early_stop_patience,
                                                         args.early_stop_queue_length,
                                                         args.early_stop_threshold_loss,
                                                         args.early_stop_threshold_perf,
                                                         args.epochs)
        result_info_dict[exp_dir]["test_perf_at_best_val_list"] = test_perf_at_best_val_list

    return result_info_dict


def run_ttest(populations, another_group):
    if isinstance(another_group, tuple) and len(another_group) == 3:
        c_mean, c_stdev, n = another_group
        _another_group = np.random.normal(c_mean, c_stdev, n).tolist()
    elif isinstance(another_group, list):
        _another_group = another_group

    t_test_ret = stats.ttest_ind(populations, _another_group)
    # t_test_ret_diff_var = stats.ttest_ind(populations, _another_group, equal_var=False)

    print(f"#: {len(populations)} vs {len(_another_group)}")
    print(f"Mean: {np.mean(populations)} +- {np.std(populations)} vs {np.mean(_another_group)} +- {np.std(_another_group)}")
    print("The t-statistic and p-value (equal variance) is {} and {}".format(*t_test_ret))


def run_ttest_for_dataset(dataset_name,
                          another_group,
                          base_path="../logs-2020-neurips/log_when_to_stop/_recorded",
                          filter_func=None):

    pops_dict = load_populations(dataset_name=dataset_name, filter_func=filter_func, base_path=base_path)
    for exp_dir, v in pops_dict.items():
        cprint(f"-- {exp_dir} --", "green")
        run_ttest(v["test_perf_at_best_val_list"], another_group)
        print()


if __name__ == '__main__':

    ppi_gat = [0.7198653396881298, 0.7191472748867286, 0.7159647244847503, 0.7118689559032562, 0.7333237352364989,
               0.7237361438822507, 0.7331175679115605, 0.7228730773714791, 0.7231993569131833, 0.7253841057513329,
               0.7190833299552745, 0.7227652702321596, 0.7248408789092468, 0.7227273583965838, 0.7108870754140959,
               0.7194123841955299, 0.7105016781003793, 0.7293866463730433, 0.7095600288505671, 0.7143654450147189,
               0.7281764155911147, 0.7054484128063543, 0.7250223099597067, 0.7198176206811265, 0.7169785956072734,
               0.7226538564581118, 0.710537560417973, 0.7214279962421152, 0.7149823092842943, 0.7219674188740423]

    run_ttest_for_dataset(dataset_name="PPI", another_group=ppi_gat,
                          filter_func=lambda _k: ("Full" not in _k and "CGAT" not in _k
                                                  and "NE" not in _k and "EV20" not in _k))
    run_ttest_for_dataset(dataset_name="Cora", another_group=(0.83, 0.007, 100),
                          filter_func=lambda _k: ("Full" not in _k and "CGAT" not in _k and "EV20" not in _k))
    run_ttest_for_dataset(dataset_name="CiteSeer", another_group=(0.725, 0.007, 100),
                          filter_func=lambda _k: ("Full" not in _k and "CGAT" not in _k and "EV20" not in _k))
    run_ttest_for_dataset(dataset_name="PubMed", another_group=(0.79, 0.004, 100),
                          filter_func=lambda _k: ("Full" not in _k and "CGAT" not in _k and "EV20" not in _k))

