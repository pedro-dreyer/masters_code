import numpy as np
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

RESULTS_NAMES = ['train_acc1', 'val_acc1', 'train_loss', 'val_loss', 'learning_rate']


class FullExperiment:
    def __init__(self, folder_path=None, experiment_list=None):

        # TODO error if both inputs are None or not None
        # if folder_path is None and experiment_list is None:

        if folder_path is not None:
            self.list = load_experiments(folder_path)
        if experiment_list is not None:
            self.list = experiment_list

        self.len = len(self.list)

    def filter_parameters(self, parameters_dict):
        new_list = []
        for experiment in self.list:
            include = True
            for parameter, value in parameters_dict.items():
                if (parameter not in experiment.parameters_dict.keys() or
                        experiment.parameters_dict[parameter] != value):
                    include = False
                    continue
            if include:
                new_list.append(experiment)
        return FullExperiment(experiment_list=new_list)

    def sort_by_parameter(self, parameter, reverse):

        # TODO parameter options

        def sort_fun(elem):
            return elem.parameters_dict[parameter]

        self.list.sort(key=sort_fun, reverse=reverse)

    def sort_by_results(self, result, metric, reverse=False, epoch_limit=None):

        #metric: 'max_mean', 'min_mean'
        #result: 'train_acc1', 'val_acc1', 'train_loss', 'val_loss', 'learning_rate'

        def sort_fun(elem):
            return elem.experiment_results(epoch_limit=epoch_limit)[result][metric]

        self.list.sort(key=sort_fun, reverse=reverse)

    def make_latex_table(self, parameter_list, result_list, result_type_list, name, precision, epoch_limit=None):

        latex_table = '\\multirow'
        latex_table += '{' + str(self.len) + '}{*}'
        latex_table += '{' + name + '}\n'

        for experiment in self.list:
            for parameter in parameter_list:
                latex_table += ' & ' + str(experiment.parameters_dict[parameter])

            experiment_summary = experiment.experiment_results(epoch_limit=epoch_limit)
            for result, result_type in zip(result_list, result_type_list):
                result_value = experiment_summary[result][result_type]
                result_str = str(result_value.round(precision))
                if len(result_str.split('.')[1]) != precision:
                    result_str += '0'

                result_str += ' ($\\pm$ '
                if 'min' in result_type:
                    sd_type = 'std_min_mean'
                else:
                    sd_type = 'std_max_mean'

                sd_value = experiment_summary[result][sd_type]
                sd_str = str(sd_value.round(precision))
                if len(sd_str.split('.')[1]) != precision:
                    sd_str += '0'

                result_str += sd_str + ')'

                latex_table += ' & ' + result_str

            latex_table += ' \\\\ \n'

        latex_table += '\\midrule \n'

        return latex_table

    def show_results(self, parameters, result_names, result_types, epoch_limit=None):
        for i_experiment, experiment in enumerate(self.list):
            print(10 * '-', 'EXPERIMENT {}'.format(i_experiment), 10 * '-')
            print(3 * '*', 'PARAMETERS', 3 * '*')
            if parameters == 'all':

                for key, value in experiment.parameters_dict.items():
                    print(key, ': ', value)
            elif parameters == 'training':
                print('training method: ', experiment.parameters_dict['training_method'])
                for key, value in experiment.training_method_parameters.items():
                    print(key, ': ', value)
            elif parameters == 'learning':
                print('learning method: ', experiment.parameters_dict['learning_method'])
                for key, value in experiment.learning_method_parameters.items():
                    print(key, ': ', value)
            elif parameters == 'T&L':
                print('training method: ', experiment.parameters_dict['training_method'])
                for key, value in experiment.training_method_parameters.items():
                    print(key, ': ', value)
                print('learning method: ', experiment.parameters_dict['learning_method'])
                for key, value in experiment.learning_method_parameters.items():
                    print(key, ': ', value)
            else:  # parameters is a list
                for parameter in parameters:
                    print(parameter, ': ', experiment.parameters_dict[parameter])

            print(3 * '*', 'RESULTS', 3 * '*')
            experiment_results = experiment.experiment_results(epoch_limit=epoch_limit)
            for result_name in result_names:
                for result_type in result_types:
                    print(result_name, experiment_results[result_name][result_type])
                    
    
    def plot_compare_graph(self, 
                           parameters_to_compare, result_to_compare, 
                           parameters_to_filter=None, values_to_filter=None,
                           ax = None, fig = None):
    
        title = 'comparing ' + parameters_to_compare[0]
        if parameters_to_filter is not None:
            title = title + ' with {} = {}'.format(parameters_to_filter, values_to_filter)

        if fig is None and ax is None:
            fig, ax = plt.subplots(1,1, constrained_layout=True)
            fig.suptitle(title)
            fig.set_size_inches((13, 5))


        ax.set_xlabel('Epochs')
        ax.set_ylabel(result_to_compare)

        for experiment in self.list:

            parameters_dict = experiment.parameters_dict
            results_dataframe = experiment.results_df

            plot_graph = True

            if parameters_to_filter is not None:
                for parameter_to_filter,value_to_filter in zip(parameters_to_filter,values_to_filter):
                    if parameters_dict[parameter_to_filter] != value_to_filter:
                        plot_graph = False
                        break

            if plot_graph == True:

                string = None
                for parameter_to_compare in parameters_to_compare:
                    if string is None:
                        string = parameter_to_compare + ' =' + str(parameters_dict[parameter_to_compare])
                    else:
                        string = string + ' and ' + parameter_to_compare + ' = ' + str(parameters_dict[parameter_to_compare])
                x = range(1,int(parameters_dict['epochs'])+1)
                y = results_dataframe[result_to_compare].mean(level=1)
                result_plot, = ax.plot(x, y ,label = string)


        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        return fig, ax
        #lines = [val_acc1_plot, train_acc1_plot, val_loss_plot, train_loss_plot, learning_rate_plot]

        #ax.legend(lines, [l.get_label() for l in lines], loc='upper left', bbox_to_anchor=(1.4, 1))
        


class SingleExperiment:

    def __init__(self, file_path):
        self.file_name = file_path.split('/')[-1]
        self.parameters_dict = get_parameters(file_path)
        self.results_df = get_results_pandas(file_path)
        self.results_ds = get_results_xarray(file_path)

        self.parameters_dict['epochs'] = int(self.parameters_dict['epochs'])
        self.parameters_dict['executions'] = int(self.parameters_dict['executions'])

        self.parameters_dict['bach_size'] = int(self.parameters_dict['batch_size'])

        self.parameters_dict['test_set_split'] = float(self.parameters_dict['test_set_split'])
        self.parameters_dict['validation_set_split'] = float(self.parameters_dict['validation_set_split'])
        self.parameters_dict['reduce_train_set'] = float(self.parameters_dict['reduce_train_set'])

        self.parameters_dict['base_seed'] = int(self.parameters_dict['base_seed'])

        training_method = self.parameters_dict['training_method']
        learning_method = self.parameters_dict['learning_method']

        self.training_method_parameters = {key: value for key, value
                                           in self.parameters_dict.items()
                                           if key.split('_')[0] == training_method}

        self.learning_method_parameters = {key: value for key, value
                                           in self.parameters_dict.items()
                                           if key.split('_')[0] == learning_method}

        for key, value in self.parameters_dict.items():
            if value == 'False':
                self.parameters_dict[key] = False
            elif value == 'True':
                self.parameters_dict[key] = True

    def experiment_results(self, epoch_limit=None):

        if epoch_limit is not None:
            tmp_results_ds = self.results_ds.drop(range(self.parameters_dict['epochs'], epoch_limit, -1), dim='epoch')
        else:
            tmp_results_ds = self.results_ds
           

        summary_dict = {}
        for result in RESULTS_NAMES:
            result_da = tmp_results_ds[result]

            mean_da = result_da.mean(dim='execution')
            std_da = result_da.std(dim='execution')
            max_da = result_da.max(dim='execution')
            min_da = result_da.min(dim='execution')

            max_mean = mean_da.max(dim='epoch').values
            max_mean_epoch = mean_da.argmax(dim='epoch').values + 1
            min_mean = mean_da.min(dim='epoch').values
            min_mean_epoch = mean_da.argmin(dim='epoch').values + 1

            summary = {'mean':         mean_da.values, 'std': std_da.values,
                       'max':          max_da.values, 'min': min_da.values,
                       'max_mean':     max_mean, 'max_mean_epoch': max_mean_epoch,
                       'min_mean':     min_mean, 'min_mean_epoch': min_mean_epoch,
                       'std_max_mean': std_da.sel(epoch=max_mean_epoch).values,
                       'std_min_mean': std_da.sel(epoch=min_mean_epoch).values}

            summary_dict[result] = summary

        return summary_dict


def get_parameters(file_path):
    with open(file_path, 'r') as f:
        parameters_dict = {}
        lines = f.readlines()
        for line in lines:
            line = line[:-1]

            # if we get to a blank line it means that the parameters have ended
            if len(line) == 0:
                break

            colon_position = line.find(':')

            if colon_position != -1:
                parameter = line[:colon_position]
                value = line[colon_position + 1:]
                try:
                    value = int(value)
                except (ValueError, TypeError) as error:
                    try:
                        value = float(value)
                    except (ValueError, TypeError)as error:
                        pass

                parameters_dict[parameter] = value

        return parameters_dict


def get_values(file_path, value_name, parameters_dict):
    n_epochs = int(parameters_dict['epochs'])
    n_executions = int(parameters_dict['executions'])
    result = np.empty((n_executions, n_epochs))
    i_execution = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()
        correct_section = False
        for line in lines:
            line = line[:-1]

            if line == value_name:
                correct_section = True
                continue

            if correct_section:
                line_list = line.split(',')
                if len(line) == 0:
                    break
                result[i_execution, :] = np.array(line_list, dtype=float)
                i_execution += 1
        return result


def get_results_xarray(file_path):
    parameters_dict = get_parameters(file_path)

    results_dict = {name: get_values(file_path, name, parameters_dict) for name in RESULTS_NAMES}
    n_executions = int(parameters_dict['executions'])
    n_epochs = int(parameters_dict['epochs'])

    results_ds = xr.Dataset()
    for key, value in results_dict.items():
        results_ds[key] = (('execution', 'epoch'), value)

    results_ds.coords['epoch'] = range(1, n_epochs + 1)
    results_ds.coords['execution'] = range(1, n_executions + 1)

    # results_df = results_ds.to_dataframe()
    # results_df = results_df.swaplevel('epoch', 'execution', axis=0)
    # results_df = results_df.sort_index(level=0)

    return results_ds


def get_results_pandas(file_path):
    parameters_dict = get_parameters(file_path)

    results_dict = {name: get_values(file_path, name, parameters_dict) for name in RESULTS_NAMES}
    n_results = len(RESULTS_NAMES)
    n_executions = int(parameters_dict['executions'])
    n_epochs = int(parameters_dict['epochs'])

    results_array = np.empty((n_results, n_executions, n_epochs))  # results, execution, epoch
    for ii, result in enumerate(results_dict.values()):
        results_array[ii] = result

    results_array = np.swapaxes(results_array, 2, 0)  # epochs, execution, results
    results_array = np.swapaxes(results_array, 0, 1)  # execution, epoch, result
    results_array = np.reshape(results_array, (n_executions * n_epochs, n_results), order='C')

    execution = [ii for ii in range(1, n_executions + 1)]
    epochs = [ii for ii in range(1, n_epochs + 1)]
    index = pd.MultiIndex.from_product([execution, epochs],
                                       names=['execution', 'epoch'])

    results_df = pd.DataFrame(results_array,
                              index=index,
                              columns=RESULTS_NAMES)

    return results_df


def load_experiments(folder_path):
    files_name = sorted(os.listdir(folder_path))
    # only load .txt files
    files_name = list(filter(lambda x: x[-3:] == 'txt', files_name))
    files_path = [folder_path + file_name for file_name in files_name]
    experiment_list = []
    for file_path in files_path:
        experiment = SingleExperiment(file_path)
        experiment_list.append(experiment)

    return experiment_list


if __name__ == '__main__':
    print(os.listdir('.'))
    folder_path_t = 'results/first_experiment/data/'
    file_name_t = 'sgd_tas-2019-01-12 18:16:22.822506.txt'
    file_path_t = folder_path_t + file_name_t
    full_experiment = FullExperiment(folder_path=folder_path_t)
    sgd_experiments = full_experiment.filter_parameters({'training_method': 'sgd'})

    # latex_table = sgd_experiments.make_latex_table('initial_learning_rate', ['val_acc1', 'val_loss'], 'test')
    # print(latex_table)

    #
    # for experiment in full_experiment.list:
    #     print(experiment.get_experiment_summary()['val_acc1']['max_mean'])
    #
    # full_experiment.sort_by_results('val_acc1', 'max_mean', reverse=False)
    # print('-------------------------')
    # for experiment in full_experiment.list:
    #     print(experiment.get_experiment_summary()['val_acc1']['max_mean'])
    # #
    # # print(full_experiment.list[0].get_experiment_summary())
