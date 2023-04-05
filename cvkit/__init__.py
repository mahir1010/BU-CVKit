from cvkit.utils import MAGIC_NUMBER

import importlib
import pkgutil

discovered_plugins = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in pkgutil.iter_modules()
    if name.startswith('cvkit_')
}

pose_estimation_plugin={}
for plugin,package in discovered_plugins.items():
    try:
        if package.CLASS == "Pose Estimation":
            pose_estimation_plugin[plugin]=package
    except Exception as ex:
        pass

print(pose_estimation_plugin)