import os
import OpenAttack

def getSubClasses(module, clss):
    ret = []
    for name in dir(module):
        try:
            if issubclass(module.__dict__[name], clss):
                ret.append(name)
        except TypeError:
            continue
    return ret

def make_attacker(path):
    addition_members = {
        "SEAAttacker": ["get_rules"],
        "UATAttacker": ["get_triggers"],
    }
    opt = "===================\nAttackers API\n===================\n\n"

    opt += "Attacker\n-----------\n\n.. autoclass:: OpenAttack.Attacker\n    :members: __call__\n\n"
    opt += "-" * 36 + "\n\n"
    
    for name in getSubClasses(OpenAttack.attackers, OpenAttack.Attacker):
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.attackers.%s(OpenAttack.Attacker)\n" % name

        members = ["__init__"] + (addition_members[name] if name in addition_members else [])
        opt += "    :members: " + (", ".join(members)) + "\n\n"
    open(path, "w", encoding="utf-8").write(opt)
    return opt

def make_attack_eval(path):
    opt = "========================\nAttackEvals API\n========================\n\n"
    opt += "AttackEval\n----------------\n\n.. autoclass:: OpenAttack.AttackEval\n    :members: __init__, eval, eval_results\n\n"
    opt += "-" * 36 + "\n\n"

    members = ["__init__", "measure", "update", "get_result", "clear", "eval", "eval_results"]
    for name in getSubClasses(OpenAttack.attack_evals, OpenAttack.AttackEval):
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.attack_evals.%s(OpenAttack.AttackEval)\n" % name
        opt += "    :members: " + (", ".join(members)) + "\n\n"
    open(path, "w", encoding="utf-8").write(opt)
    return opt

def make_classifier(path):
    addition_members = {
        "PytorchClassifier": ["to"],
        "TensorflowClassifier": ["to"],
    }
    skip_list = {"ClassifierBase"}

    opt = "===================\nClassifiers API\n===================\n\n"
    opt += "Classifier\n-----------------\n\n.. autoclass:: OpenAttack.Classifier\n    :members:\n\n"
    opt += "-" * 36 + "\n\n"
    
    for name in getSubClasses(OpenAttack.classifiers, OpenAttack.Classifier):
        if name in skip_list:
            continue

        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.classifiers.%s(OpenAttack.Classifier)\n" % name

        members = ["__init__"] + (addition_members[name] if name in addition_members else [])
        opt += "    :members: " + (", ".join(members)) + "\n\n"
    open(path, "w", encoding="utf-8").write(opt)
    return opt
    
def make_data_manager(path):
    opt = "===================\nDataManager API\n===================\n\n.. autoclass:: OpenAttack.DataManager\n    :members:"
    open(path, "w", encoding="utf-8").write(opt)
    return opt

def make_data(path):
    opt = "=====================\ndata\n=====================\n\n"
    opt += ".. _label-apis-data:\n\n"

    import pkgutil
    for data in pkgutil.iter_modules(OpenAttack.data.__path__):
        data = data.module_finder.find_loader(data.name)[0].load_module()
        if hasattr(data, "NAME") and (data.NAME in OpenAttack.DataManager.AVAILABLE_DATAS):
            opt += data.NAME + "\n" + ("-" * (2 + len(data.NAME))) + "\n\n"
            opt += ".. py:data:: " + data.NAME + "\n\n"
            opt += "    .. automodule:: OpenAttack.data." + data.__name__ + "\n\n"
    open(path, "w", encoding="utf-8").write(opt)
    return opt

def make_metric(path):
    opt = "==================\nMetric API\n==================\n\n"
    
    members = ["__init__", "__call__"]

    for name in OpenAttack.metric.__dir__():
        if type(OpenAttack.metric.__dict__[name]) is type:
            opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
            opt += ".. autoclass:: OpenAttack.metric." + name + "\n"
            opt += "    :members: " + (", ".join(members)) + "\n\n"
    open(path, "w", encoding="utf-8").write(opt)
    return opt

def make_substitute(path):
    opt = "======================\nSubstitutes API\n======================\n\n"
    bases = ["WordSubstitute", "CharSubstitute"]
    for name in bases:
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.substitutes.base." + name + "\n"
        opt += "    :members: __call__\n\n"
    opt += "-" * 36 + "\n\n"

    for name in getSubClasses(OpenAttack.substitutes, OpenAttack.Substitute):
        cls = OpenAttack.substitutes.__dict__[name]
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.substitutes.%s(OpenAttack.substitutes.%s)\n" % (name, cls.__base__.__name__)
        opt += "    :members:\n\n"
    open(path, "w", encoding="utf-8").write(opt)
    return opt

def make_text_processor(path):
    opt = "========================\nTextProcessors API\n========================\n\n"
    opt += "TextProcessor\n--------------------\n\n.. autoclass:: OpenAttack.TextProcessor\n    :members:\n\n"
    opt += "-" * 36 + "\n\n"
    for name in getSubClasses(OpenAttack.text_processors, OpenAttack.TextProcessor):
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        opt += ".. autoclass:: OpenAttack.text_processors.%s(OpenAttack.TextProcessor)\n" % name
        opt += "    :members:\n\n"
    open(path, "w", encoding="utf-8").write(opt)
    return opt

def make_utils(path):
    opt = "=====================\nutils API\n=====================\n\n"
    for name in OpenAttack.utils.__dir__():
        if name.startswith("__"):
            continue
        obj = OpenAttack.utils.__dict__[name]
        if type(obj).__name__ == "module":
            continue
        opt += name + "\n" + ("-" * (2 + len(name))) + "\n\n"
        if type(obj).__name__  == "function":
            opt += ".. autofunction:: OpenAttack.utils." + name + "\n\n"
        else:
            opt += ".. autoclass:: OpenAttack.utils." + name + "\n"
            opt += "    :members:\n\n"
    open(path, "w", encoding="utf-8").write(opt)
    return opt

def main(path):
    make_attacker(os.path.join(path, "attacker.rst"))
    make_attack_eval(os.path.join(path, "attack_eval.rst"))
    make_classifier(os.path.join(path, "classifier.rst"))
    make_data_manager(os.path.join(path, "data_manager.rst"))
    make_data(os.path.join(path, "data.rst"))
    make_metric(os.path.join(path, "metric.rst"))
    make_substitute(os.path.join(path, "substitute.rst"))
    make_text_processor(os.path.join(path, "text_processor.rst"))
    make_utils(os.path.join(path, "utils.rst"))

if __name__ == "__main__":
    import sys
    path = os.path.abspath(sys.argv[1])
    main(path)