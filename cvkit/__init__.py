import importlib
import pkgutil
from inspect import isclass
from os.path import exists, join, dirname

from cvkit.utils import MAGIC_NUMBER


class MetaProcessor:
    def __init__(self, plugin_name, processor_type, processor_class):
        self.plugin_name = plugin_name
        self.processor_type = processor_type
        self.processor_class = processor_class

    def __str__(self):
        return f'{self.plugin_name} - {self.processor_type} - {self.processor_class.PROCESSOR_NAME}'


discovered_pe_processors = {}

discovered_plugins = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in pkgutil.iter_modules()
    if name.startswith('cvkit_')
}

pose_estimation_plugin = ['cvkit']
for plugin, package in discovered_plugins.items():
    try:
        if package.CLASS == "Pose Estimation":
            pose_estimation_plugin.append(plugin)

    except Exception as ex:
        print(ex)
        pass

abstract_processor_class = getattr(importlib.import_module(f'cvkit.pose_estimation.processors.processor_interface'),
                                   'Processor')


def register_processor(processor: MetaProcessor):
    if processor.processor_class.PROCESSOR_ID not in discovered_pe_processors:
        discovered_pe_processors[processor.processor_class.PROCESSOR_ID] = processor


for plugin in pose_estimation_plugin:
    # Look for generative processors
    generative_processors_dir = join(dirname(pkgutil.get_loader(plugin).get_filename()), 'pose_estimation',
                                     'processors', 'generative')
    if exists(generative_processors_dir):
        for _, name, _ in pkgutil.iter_modules(path=[generative_processors_dir]):
            module = importlib.import_module(f'{plugin}.pose_estimation.processors.generative.{name}')
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isclass(attribute) and issubclass(attribute,
                                                     abstract_processor_class) and attribute != abstract_processor_class:
                    register_processor(MetaProcessor(plugin, 'generative', attribute))
    # Look for filter processors
    filter_processors_dir = join(dirname(pkgutil.get_loader(plugin).get_filename()), 'pose_estimation', 'processors',
                                 'filter')
    if exists(filter_processors_dir):
        for _, name, _ in pkgutil.iter_modules(path=[filter_processors_dir]):
            module = importlib.import_module(f'{plugin}.pose_estimation.processors.filter.{name}')
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isclass(attribute) and issubclass(attribute,
                                                     abstract_processor_class) and attribute != abstract_processor_class:
                    register_processor(MetaProcessor(plugin, 'filter', attribute))
    # Look for Util processors
    util_processors_dir = join(dirname(pkgutil.get_loader(plugin).get_filename()), 'pose_estimation', 'processors',
                               'util')
    if exists(util_processors_dir):
        for _, name, _ in pkgutil.iter_modules(path=[util_processors_dir]):
            module = importlib.import_module(f'{plugin}.pose_estimation.processors.util.{name}')
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isclass(attribute) and issubclass(attribute,
                                                     abstract_processor_class) and attribute != abstract_processor_class:
                    register_processor(MetaProcessor(plugin, 'util', attribute))


def get_processor_class(processor_id):
    meta_class = discovered_pe_processors.get(processor_id, None)
    if meta_class:
        return meta_class.processor_class


def verify_installed_processor(processor_id):
    return processor_id in discovered_pe_processors
